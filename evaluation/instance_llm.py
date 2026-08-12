"""
LLM 기반 인스턴스 그룹핑 (2단계 파이프라인).

- 입력: UV-Net 면 단위 예측(.seg) + 면 기하 속성(.attrs.json)
- 처리: 면별 [예측 라벨, 곡면 타입, 면적, 중심, 축, 반경, 인접]을 텍스트로
  직렬화해 LLM에 질의 → 인스턴스 JSON을 받는다.
- 평가: instance_baseline.py와 동일한 지표(P/R/F1, mean matched IoU).

API: OpenAI 호환 chat completions 엔드포인트(OpenAI, vLLM, OpenRouter, Ollama 등).
API 키는 환경변수(config의 api_key_env에 지정한 이름)에서 읽는다.

실행: python instance_llm.py  (설정: instance_llm_config.json)
"""
import json
import os
import re
import sys
import time

import requests
from tqdm import tqdm

module_path = os.path.join(os.getcwd(), 'src')
sys.path.append(module_path)

from src.config import Config
from src.instance_common import (
    load_seg,
    load_face_attrs,
    load_gt_instances,
    load_class_names,
    group_by_connected_components,
    serialize_part,
    evaluate_part,
    aggregate_metrics,
)

SYSTEM_PROMPT = """You are an expert in CAD machining feature recognition on B-Rep models.

You are given the faces of one CAD part. Each line describes one face:
- pred: the machining feature class predicted for this face by an upstream model (may contain occasional errors)
- surf: surface type (plane / cylinder / cone / sphere / torus / freeform)
- area, centroid, axis (surface normal for planes, rotation axis otherwise), radius
- bottom=1 (optional): this face is a feature bottom face
- adj: adjacent face ids, each annotated with the shared edge's convexity:
  :cc = concave, :cv = convex, :sm = smooth (tangent-continuous)
  Concave edges are the strongest signal of a feature cavity boundary — faces joined
  by a concave edge usually belong to the same feature instance, while convex edges
  typically separate feature faces from stock faces.

Task: group faces into machining feature INSTANCES. One instance is one occurrence of a
feature (e.g., a part with two separate holes has two instances, each with its own faces).

Rules:
1. Every face must belong to at least one instance, except faces listed as stock/base
   classes: {exclude_desc}. Do not include those faces in any instance.
2. Faces of the same instance are normally connected through adjacency.
3. Use geometry to separate touching or INTERSECTING instances of the same type.
   Each feature type has a geometric invariant that defines one instance:
   - slot/passage: a pair of OPPOSITE PARALLEL side walls sharing one sweep axis
   - hole: COAXIAL cylindrical faces with the same radius
   - pocket: one bottom face enclosed by a loop of side walls
   Example: two identical slots crossing in a "+" shape share one merged bottom face;
   group the side walls by their normal direction (one parallel-wall pair per slot)
   into TWO instances, and put the shared bottom face in BOTH instances.
4. A face may appear in multiple instances ONLY when features genuinely intersect and
   share that face (e.g., the merged bottom at a slot crossing). Otherwise assign each
   face to exactly one instance.
5. INTERRUPTED instances: when one feature is cut through by an intersecting feature,
   the faces of the interrupted feature may form two or more disconnected groups
   (e.g., the two collinear halves of a shallow slot severed by a deeper crossing slot,
   or a pocket whose walls are split into two sides by a through slot passing across it).
   Do NOT output such fragments as separate instances. Unify fragments into ONE instance
   when they share the feature's geometric invariant: collinear sweep axis and equal
   width/depth for slots; one common wall loop and floor for pockets. The number of
   instances should equal the number of distinct machining operations, not the number
   of connected face groups.
6. The upstream per-face predictions are mostly reliable; only override a face's class
   when geometry and its neighborhood make the prediction clearly inconsistent.
7. The "type" of an instance is the class id (integer) shared by its faces.

Answer with ONLY a JSON object, no other text:
{{"instances": [{{"type": <class_id_int>, "faces": [<face_id_int>, ...]}}, ...]}}"""


def build_messages(face_text, exclude_labels, class_names):
    if exclude_labels:
        if class_names:
            exclude_desc = ", ".join(
                f"{class_names.get(l, 'class_' + str(l))}({l})" for l in exclude_labels)
        else:
            exclude_desc = ", ".join(f"class_{l}" for l in exclude_labels)
    else:
        exclude_desc = "(none)"

    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(exclude_desc=exclude_desc)},
        {"role": "user", "content": face_text},
    ]


def call_llm(config, messages):
    """OpenAI 호환 chat completions 호출. 실패 시 재시도."""
    api_key = os.environ.get(config.api_key_env, "")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": config.model,
        "messages": messages,
    }
    # reasoning 계열(gpt-5.x, o-시리즈)은 temperature 고정 + max_completion_tokens 사용
    if not getattr(config, "omit_temperature", False):
        payload["temperature"] = config.temperature
    if getattr(config, "use_max_completion_tokens", False):
        payload["max_completion_tokens"] = config.max_tokens
    else:
        payload["max_tokens"] = config.max_tokens

    last_error = None
    for attempt in range(config.n_retries + 1):
        try:
            response = requests.post(
                config.api_base.rstrip("/") + "/chat/completions",
                headers=headers, json=payload, timeout=config.timeout_seconds,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            last_error = e
            if attempt < config.n_retries:
                time.sleep(2 ** attempt)

    raise RuntimeError(f"LLM call failed after {config.n_retries + 1} attempts: {last_error}")


def parse_instances(response_text, num_faces):
    """응답에서 인스턴스 JSON을 추출·검증. 존재하지 않는 면 id는 제거(환각 방지)."""
    text = response_text.strip()

    # 코드펜스 제거
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    # 가장 바깥 중괄호 구간 추출
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in response.")

    data = json.loads(text[start:end + 1])
    raw_instances = data["instances"]

    instances = []
    for inst in raw_instances:
        faces = sorted({int(f) for f in inst["faces"] if 0 <= int(f) < num_faces})
        if faces:
            instances.append({"type": int(inst["type"]), "faces": faces})
    return instances


def find_gt_instances(stem, config, adjacency):
    """정답 인스턴스 로드. .inst.json 우선, 없으면 정답 .seg에서 CC로 유도."""
    inst_path = os.path.join(config.gt_instance_path, f"{stem}.inst.json")
    if os.path.exists(inst_path):
        return load_gt_instances(inst_path)

    gt_seg_path = os.path.join(config.gt_seg_path, f"{stem}.seg")
    if os.path.exists(gt_seg_path):
        gt_labels = load_seg(gt_seg_path)
        return group_by_connected_components(gt_labels, adjacency, config.exclude_labels)

    return None


def run(config):
    class_names = load_class_names(getattr(config, "class_names_path", None))
    pred_seg_files = sorted(f for f in os.listdir(config.pred_seg_path) if f.endswith(".seg"))

    if getattr(config, "max_parts", 0) > 0:
        pred_seg_files = pred_seg_files[:config.max_parts]

    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    raw_dir = os.path.join(config.output_dir, "raw_responses")
    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)

    part_results = []
    n_skipped, n_parse_failed = 0, 0

    for seg_file in tqdm(pred_seg_files):
        stem = os.path.splitext(seg_file)[0]

        attrs_path = os.path.join(config.attrs_path, f"{stem}.attrs.json")
        if not os.path.exists(attrs_path):
            n_skipped += 1
            continue

        attrs = load_face_attrs(attrs_path)
        pred_labels = load_seg(os.path.join(config.pred_seg_path, seg_file))

        if len(pred_labels) != attrs["num_faces"]:
            print(f"[warn] {stem}: face count mismatch. Skipping.")
            n_skipped += 1
            continue

        gt_instances = find_gt_instances(stem, config, attrs["adjacency"])
        if gt_instances is None:
            n_skipped += 1
            continue

        # skip_existing: 이미 출력이 있으면 API 호출 없이 기존 예측을 평가에 포함
        # (쿼터 제한으로 여러 날에 나눠 실행할 때 이어받기용)
        out_path = os.path.join(config.output_dir, f"{stem}.inst.json")
        if getattr(config, "skip_existing", False) and os.path.exists(out_path):
            with open(out_path, encoding="UTF8") as f:
                pred_instances = json.load(f)
            part_results.append(evaluate_part(pred_instances, gt_instances, config.iou_threshold))
            continue

        # 직렬화 → LLM 질의 → 파싱
        # bottom_path가 설정돼 있으면 면별 바닥면 힌트(0/1 리스트 JSON)를 함께 직렬화.
        # 주의: GT에서 유도한 바닥면 라벨을 쓰면 정보 누설 — 예측기 출력일 때만 공정.
        bottom_flags = None
        bottom_dir = getattr(config, "bottom_path", None)
        if bottom_dir:
            bottom_file = os.path.join(bottom_dir, f"{stem}.json")
            if os.path.exists(bottom_file):
                with open(bottom_file, encoding="UTF8") as bf:
                    bottom_flags = json.load(bf)

        face_text = serialize_part(attrs, pred_labels, class_names,
                                   bottom_flags=bottom_flags)
        messages = build_messages(face_text, config.exclude_labels, class_names)

        try:
            response_text = call_llm(config, messages)
        except RuntimeError as e:
            print(f"[error] {stem}: {e}")
            n_skipped += 1
            continue

        # 무료 티어 등 RPM 제한 대응: 호출 간 간격 (config에 없으면 0)
        interval = getattr(config, "request_interval_seconds", 0)
        if interval:
            time.sleep(interval)

        with open(os.path.join(raw_dir, f"{stem}.txt"), 'w', encoding='UTF8') as f:
            f.write(response_text)

        try:
            pred_instances = parse_instances(response_text, attrs["num_faces"])
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            print(f"[warn] {stem}: parse failed ({e}). Falling back to rule grouping.")
            n_parse_failed += 1
            # 파싱 실패 시 룰 그룹핑으로 폴백 (평가 누락 방지)
            pred_instances = group_by_connected_components(
                pred_labels, attrs["adjacency"], config.exclude_labels)

        with open(os.path.join(config.output_dir, f"{stem}.inst.json"), 'w', encoding='UTF8') as f:
            json.dump(pred_instances, f, ensure_ascii=False)

        part_results.append(evaluate_part(pred_instances, gt_instances, config.iou_threshold))

    metrics = aggregate_metrics(part_results)
    metrics["n_parse_failed"] = n_parse_failed
    metrics["model"] = config.model

    print()
    print(f"=== LLM instance grouping ({config.model}) ===")
    print(f"Parts evaluated : {metrics['n_parts']} (skipped: {n_skipped}, parse failed: {n_parse_failed})")
    print(f"Precision       : {metrics['precision']:.4f}")
    print(f"Recall          : {metrics['recall']:.4f}")
    print(f"F1              : {metrics['f1']:.4f}")
    print(f"Mean matched IoU: {metrics['mean_matched_iou']:.4f}")

    with open(os.path.join(config.output_dir, "llm_metrics.json"), 'w', encoding='UTF8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Saved metrics to {os.path.join(config.output_dir, 'llm_metrics.json')}")


def main():
    config = Config()
    # 병렬 실행용: 환경변수로 설정 파일 경로 지정 가능 (기본값 유지)
    config.load(os.environ.get("INSTANCE_LLM_CONFIG", "instance_llm_config.json"))
    run(config)


if __name__ == "__main__":
    main()

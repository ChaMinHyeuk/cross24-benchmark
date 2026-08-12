"""
인스턴스 그룹핑 파이프라인 공통 유틸.

- UV-Net(1단계)의 면 단위 예측(.seg)과 면 기하 속성(.attrs.json)을 읽어
  인스턴스 그룹핑(2단계)의 입력을 만들고, 예측 인스턴스를 평가한다.
- 인스턴스 표현: {"type": <클래스 id(int)>, "faces": [<면 id(int)>, ...]}
"""
import json
import os
from collections import defaultdict, deque


# ---------------------------------------------------------------------------
# 로더
# ---------------------------------------------------------------------------

def load_seg(seg_file_path):
    """.seg 파일(한 줄에 면 하나의 라벨)을 읽어 int 리스트로 반환."""
    with open(seg_file_path, "r") as f:
        return [int(line.strip()) for line in f if line.strip() != ""]


def load_face_attrs(attrs_file_path):
    """extract_face_attrs.py가 생성한 면 기하 속성 JSON을 읽는다."""
    with open(attrs_file_path, encoding="UTF8") as f:
        return json.load(f)


def load_gt_instances(inst_file_path):
    """정답 인스턴스 파일(.inst.json)을 읽는다.

    형식: [{"type": int, "faces": [int, ...]}, ...]
    MFInstSeg 등 원본 데이터셋의 라벨은 별도 변환 스크립트로 이 형식에 맞춘다.
    """
    with open(inst_file_path, encoding="UTF8") as f:
        instances = json.load(f)
    return [{"type": int(inst["type"]), "faces": sorted(int(i) for i in inst["faces"])}
            for inst in instances]


def load_class_names(class_names_file_path):
    """{클래스 id(str): 이름} JSON을 {int: str}로 읽는다. 파일이 없으면 None."""
    if class_names_file_path and os.path.exists(class_names_file_path):
        with open(class_names_file_path, encoding="UTF8") as f:
            raw = json.load(f)
        return {int(k): v for k, v in raw.items()}
    return None


# ---------------------------------------------------------------------------
# 룰 기반 그룹핑 (베이스라인 & 정답 유도용)
# ---------------------------------------------------------------------------

def group_by_connected_components(labels, adjacency, exclude_labels=()):
    """같은 라벨을 가진 인접 면들을 connected component로 묶어 인스턴스를 만든다.

    labels        : 면별 클래스 id 리스트 (index = 면 id)
    adjacency     : [[a, b], ...] 면 인접 쌍 리스트
    exclude_labels: 인스턴스로 묶지 않을 라벨(스톡/모재 면 등)

    주의: 서로 교차/접촉하는 동일 타입 피처는 하나로 합쳐지는 한계가 있다.
    (이 한계가 LLM 그룹핑이 개선해야 할 지점)
    """
    exclude = set(exclude_labels)
    neighbors = defaultdict(set)
    for a, b in adjacency:
        neighbors[a].add(b)
        neighbors[b].add(a)

    visited = set()
    instances = []
    for face_id, label in enumerate(labels):
        if label in exclude or face_id in visited:
            continue
        # BFS로 같은 라벨의 인접 면을 모두 수집
        component = []
        queue = deque([face_id])
        visited.add(face_id)
        while queue:
            current = queue.popleft()
            component.append(current)
            for nxt in neighbors[current]:
                if nxt not in visited and nxt < len(labels) and labels[nxt] == label:
                    visited.add(nxt)
                    queue.append(nxt)
        instances.append({"type": label, "faces": sorted(component)})

    return instances


# ---------------------------------------------------------------------------
# 평가 (인스턴스 매칭 기반 P/R/F1)
# ---------------------------------------------------------------------------

def instance_iou(faces_a, faces_b):
    """두 인스턴스(면 id 집합)의 IoU."""
    set_a, set_b = set(faces_a), set(faces_b)
    union = len(set_a | set_b)
    return len(set_a & set_b) / union if union > 0 else 0.0


def match_instances(pred_instances, gt_instances, iou_threshold=0.5):
    """예측/정답 인스턴스를 IoU 내림차순 greedy 매칭.

    TP 조건: 클래스 일치 + IoU >= iou_threshold. 각 인스턴스는 한 번만 매칭.
    반환: (matches, matched_ious)  — matches = [(pred_idx, gt_idx), ...]
    """
    candidates = []
    for p_idx, pred in enumerate(pred_instances):
        for g_idx, gt in enumerate(gt_instances):
            if pred["type"] != gt["type"]:
                continue
            iou = instance_iou(pred["faces"], gt["faces"])
            if iou >= iou_threshold:
                candidates.append((iou, p_idx, g_idx))

    candidates.sort(reverse=True)
    used_pred, used_gt = set(), set()
    matches, matched_ious = [], []
    for iou, p_idx, g_idx in candidates:
        if p_idx in used_pred or g_idx in used_gt:
            continue
        used_pred.add(p_idx)
        used_gt.add(g_idx)
        matches.append((p_idx, g_idx))
        matched_ious.append(iou)

    return matches, matched_ious


def evaluate_part(pred_instances, gt_instances, iou_threshold=0.5):
    """파트 하나에 대한 카운트 반환: {"tp", "n_pred", "n_gt", "matched_ious"}"""
    matches, matched_ious = match_instances(pred_instances, gt_instances, iou_threshold)
    return {
        "tp": len(matches),
        "n_pred": len(pred_instances),
        "n_gt": len(gt_instances),
        "matched_ious": matched_ious,
    }


def aggregate_metrics(part_results):
    """파트별 카운트를 모아 micro precision / recall / F1 / mean matched IoU 계산."""
    tp = sum(r["tp"] for r in part_results)
    n_pred = sum(r["n_pred"] for r in part_results)
    n_gt = sum(r["n_gt"] for r in part_results)
    all_ious = [iou for r in part_results for iou in r["matched_ious"]]

    precision = tp / n_pred if n_pred > 0 else 0.0
    recall = tp / n_gt if n_gt > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    mean_iou = sum(all_ious) / len(all_ious) if all_ious else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_matched_iou": mean_iou,
        "tp": tp,
        "n_pred": n_pred,
        "n_gt": n_gt,
        "n_parts": len(part_results),
    }


# ---------------------------------------------------------------------------
# LLM 입력 직렬화
# ---------------------------------------------------------------------------

def _fmt_vec(vec, ndigits=2):
    return "[" + ", ".join(f"{v:.{ndigits}f}" for v in vec) + "]"


_CONVEXITY_MARK = {"concave": "cc", "convex": "cv", "smooth": "sm", "unknown": "?"}


def serialize_part(face_attrs, pred_labels, class_names=None, pred_probs=None,
                   bottom_flags=None):
    """면 속성 + UV-Net 예측을 LLM 프롬프트용 텍스트로 직렬화.

    face_attrs  : extract_face_attrs.py 출력 dict
    pred_labels : UV-Net의 면별 예측 클래스 id 리스트
    class_names : {id: 이름} (없으면 "class_<id>")
    pred_probs  : 면별 top-k [(클래스 id, 확률), ...] 리스트 (선택)
    bottom_flags: 면별 바닥면 여부 0/1 리스트 (선택 — 예측기나 별도 힌트가 있을 때만)

    adjacency_convexity가 있으면 인접 면 id에 엣지 볼록성을 마킹한다:
      adj=[3:cc, 5:cv, 8:sm]  (cc=concave, cv=convex, sm=smooth)
    오목(cc) 엣지는 피처 캐비티 경계의 강한 신호다 (AAGNet gAAG 엣지 속성 차용).
    """
    def name_of(label):
        if class_names and label in class_names:
            return f"{class_names[label]}({label})"
        return f"class_{label}"

    convexity = {}
    for entry in face_attrs.get("adjacency_convexity", []):
        a, b, conv = entry[0], entry[1], entry[2]
        mark = _CONVEXITY_MARK.get(conv, "?")
        convexity[(a, b)] = mark
        convexity[(b, a)] = mark

    neighbors = defaultdict(set)
    for a, b in face_attrs["adjacency"]:
        neighbors[a].add(b)
        neighbors[b].add(a)

    lines = []
    for face in face_attrs["faces"]:
        fid = face["id"]
        parts = [
            f"face_{fid}:",
            f"pred={name_of(pred_labels[fid])}",
            f"surf={face['surface_type']}",
            f"area={face['area']:.2f}",
            f"centroid={_fmt_vec(face['centroid'])}",
        ]
        if face.get("axis") is not None:
            parts.append(f"axis={_fmt_vec(face['axis'])}")
        if face.get("radius") is not None:
            parts.append(f"radius={face['radius']:.2f}")
        if bottom_flags is not None and fid < len(bottom_flags) and bottom_flags[fid]:
            parts.append("bottom=1")
        if pred_probs is not None:
            topk = ", ".join(f"{name_of(c)}:{p:.2f}" for c, p in pred_probs[fid])
            parts.append(f"probs=({topk})")
        if convexity:
            adj = ", ".join(f"{n}:{convexity.get((fid, n), '?')}" for n in sorted(neighbors[fid]))
            parts.append(f"adj=[{adj}]")
        else:
            parts.append("adj=" + str(sorted(neighbors[fid])))
        lines.append(" ".join(parts))

    return "\n".join(lines)

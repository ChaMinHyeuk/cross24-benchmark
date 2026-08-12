"""
인스턴스 그룹핑 룰 베이스라인 + 평가.

- 예측: UV-Net 면 단위 예측(.seg)에 대해 "같은 라벨 + 인접 → 같은 인스턴스"
  (connected component) 룰로 인스턴스를 만든다.
- 정답: gt_instance_path의 <stem>.inst.json이 있으면 사용하고,
  없으면 정답 .seg에서 같은 룰로 유도한다(교차 피처가 없는 데이터셋 전용).
- LLM 그룹핑(instance_llm.py)과 동일한 지표로 평가하므로 직접 비교 가능.

실행: python instance_baseline.py  (설정: instance_baseline_config.json)
"""
import json
import os
import sys

from tqdm import tqdm

module_path = os.path.join(os.getcwd(), 'src')
sys.path.append(module_path)

from src.config import Config
from src.instance_common import (
    load_seg,
    load_face_attrs,
    load_gt_instances,
    group_by_connected_components,
    evaluate_part,
    aggregate_metrics,
)


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
    pred_seg_files = [f for f in os.listdir(config.pred_seg_path) if f.endswith(".seg")]

    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    part_results = []
    n_skipped = 0

    for seg_file in tqdm(pred_seg_files):
        stem = os.path.splitext(seg_file)[0]

        attrs_path = os.path.join(config.attrs_path, f"{stem}.attrs.json")
        if not os.path.exists(attrs_path):
            n_skipped += 1
            continue

        attrs = load_face_attrs(attrs_path)
        pred_labels = load_seg(os.path.join(config.pred_seg_path, seg_file))

        if len(pred_labels) != attrs["num_faces"]:
            print(f"[warn] {stem}: face count mismatch "
                  f"(seg {len(pred_labels)} vs attrs {attrs['num_faces']}). Skipping.")
            n_skipped += 1
            continue

        gt_instances = find_gt_instances(stem, config, attrs["adjacency"])
        if gt_instances is None:
            n_skipped += 1
            continue

        # 룰 베이스라인: 예측 라벨에 대한 connected component
        pred_instances = group_by_connected_components(
            pred_labels, attrs["adjacency"], config.exclude_labels
        )

        # 예측 인스턴스 저장 (LLM 결과와 비교/분석용)
        with open(os.path.join(config.output_dir, f"{stem}.inst.json"), 'w', encoding='UTF8') as f:
            json.dump(pred_instances, f, ensure_ascii=False)

        part_results.append(evaluate_part(pred_instances, gt_instances, config.iou_threshold))

    metrics = aggregate_metrics(part_results)

    print()
    print("=== Rule baseline (connected components) ===")
    print(f"Parts evaluated : {metrics['n_parts']} (skipped: {n_skipped})")
    print(f"Precision       : {metrics['precision']:.4f}")
    print(f"Recall          : {metrics['recall']:.4f}")
    print(f"F1              : {metrics['f1']:.4f}")
    print(f"Mean matched IoU: {metrics['mean_matched_iou']:.4f}")

    with open(os.path.join(config.output_dir, "baseline_metrics.json"), 'w', encoding='UTF8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Saved metrics to {os.path.join(config.output_dir, 'baseline_metrics.json')}")


def main():
    config = Config()
    config.load("instance_baseline_config.json")
    run(config)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""전체 실험 결과 통합 집계 — 논문 표 생성용.

mfinst30(NNConv seg 입력) / cross24(GT seg 입력) 두 벤치마크에서:
- 룰 CC 베이스라인 (직접 계산)
- 학습 기반 4종 (AAGNet-official, pair, disc, obj-gnn — 저장된 출력 재채점)
- LLM 매트릭스 12종 (저장된 출력 재채점)
출력: D:\\BRepNet_exp\\matrix\\all_results.json + 콘솔 표
"""
import json
import os
import sys
from collections import Counter

REPO = r"c:\Users\차민혁\OneDrive - 고등기술연구원\바탕 화면\BRepNet_share-master"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))
from src.instance_common import (load_gt_instances, load_seg, evaluate_part,
                                 aggregate_metrics, group_by_connected_components,
                                 load_face_attrs)

DATA = r"D:\BRepNet_exp\dataset\MFInstSeg"
NNCONV_SEG = r"D:\BRepNet_exp\runs\output_20260812_153230\test_output\seg"
CROSS = r"D:\BRepNet_exp\cross_exp"
STOCK = 24


def bench_stems(bench):
    if bench == "mfinst30":
        return sorted(f[:-4] for f in os.listdir(NNCONV_SEG) if f.endswith(".seg"))[:30]
    return sorted(f[:-10] for f in os.listdir(os.path.join(CROSS, "inst")))


def gt_of(bench, stem):
    d = os.path.join(DATA, "inst") if bench == "mfinst30" else os.path.join(CROSS, "inst")
    return load_gt_instances(os.path.join(d, stem + ".inst.json"))


def score_pred_dir(bench, pred_dir, suffix=".inst.json"):
    rows = []
    n_missing = 0
    for stem in bench_stems(bench):
        p = os.path.join(pred_dir, stem + suffix)
        if not os.path.exists(p):
            n_missing += 1
            continue
        pred = json.load(open(p, encoding="UTF8"))
        rows.append(evaluate_part(pred, gt_of(bench, stem), 0.5))
    if not rows:
        return None
    m = aggregate_metrics(rows)
    m["n_missing"] = n_missing
    return m


def score_cc(bench):
    rows = []
    for stem in bench_stems(bench):
        if bench == "mfinst30":
            seg = load_seg(os.path.join(NNCONV_SEG, stem + ".seg"))
            attrs = load_face_attrs(os.path.join(r"D:\BRepNet_exp\attrs", stem + ".attrs.json"))
        else:
            seg = load_seg(os.path.join(CROSS, "seg", stem + ".seg"))
            attrs = load_face_attrs(os.path.join(CROSS, "attrs", stem + ".attrs.json"))
        pred = group_by_connected_components(seg, attrs["adjacency"], (STOCK,))
        rows.append(evaluate_part(pred, gt_of(bench, stem), 0.5))
    return aggregate_metrics(rows)


def main():
    results = {}

    # 룰 베이스라인
    for bench in ("mfinst30", "cross24"):
        results.setdefault("rule-CC", {})[bench] = score_cc(bench)

    # 학습 기반
    learned = {
        "AAGNet-official": {
            "mfinst30": r"D:\BRepNet_exp\aagnet_official\mfinst30",
            "cross24": r"D:\BRepNet_exp\aagnet_official\cross24"},
        "AAG-classic": {
            "mfinst30": r"D:\BRepNet_exp\inst_head\aagclassic_mfinst30",
            "cross24": r"D:\BRepNet_exp\inst_head\aagclassic_cross24"},
        "edge-cut": {
            "mfinst30": r"D:\BRepNet_exp\inst_head\edgecut_mfinst30",
            "cross24": r"D:\BRepNet_exp\inst_head\edgecut_cross24"},
        "disc-embedding": {
            "mfinst30": r"D:\BRepNet_exp\inst_head\disc_mfinst30",
            "cross24": r"D:\BRepNet_exp\inst_head\disc_cross24"},
    }
    for name, dirs in learned.items():
        for bench, d in dirs.items():
            if os.path.isdir(d):
                results.setdefault(name, {})[bench] = score_pred_dir(bench, d)

    # LLM 매트릭스
    matrix = r"D:\BRepNet_exp\matrix"
    for slug in sorted(os.listdir(matrix)):
        mdir = os.path.join(matrix, slug)
        if not os.path.isdir(mdir):
            continue
        for bench in ("mfinst30", "cross24"):
            d = os.path.join(mdir, bench)
            if os.path.isdir(d):
                results.setdefault("LLM:" + slug, {})[bench] = score_pred_dir(bench, d)

    with open(os.path.join(matrix, "all_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"{'방법':<26} {'mfinst30 F1':>12} {'cross24 F1':>12}")
    print("-" * 54)
    for name, r in sorted(results.items(),
                          key=lambda kv: -(kv[1].get("mfinst30") or {}).get("f1", 0)):
        m30 = (r.get("mfinst30") or {}).get("f1")
        c24 = (r.get("cross24") or {}).get("f1")
        fmt = lambda v: f"{v:.4f}" if v is not None else "—"
        print(f"{name:<26} {fmt(m30):>12} {fmt(c24):>12}")


if __name__ == "__main__":
    main()

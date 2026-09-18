# -*- coding: utf-8 -*-
"""MFInstSeg 테스트 파트(attrs가 있는 500파트)의 GT 인스턴스 3층 분류 — API 호출 없음.

tier A (CC-trivial) : 같은 클래스의 다른 GT 인스턴스와 인접하지 않음 → 연결성분 규칙으로 자명
tier B (touching)   : 같은 클래스의 다른 GT 인스턴스와 인접 → 연결성분 규칙이 병합
tier C (disconnected): 인스턴스 자신이 인접 그래프상 2개 이상의 연결성분으로 끊김
(B와 C는 중복 가능; C를 우선 표기)

추가로 GT 라벨 + CC 규칙의 상한 F1, mfinst30 내 분포, 'hard' 파트 목록(tier B/C 인스턴스 ≥1)을 출력.
출력: D:\\BRepNet_exp\\matrix\\mfinstseg_hard_subset.json
"""
import json, os, sys
from collections import Counter, defaultdict
REPO = r"c:\Users\차민혁\OneDrive - 고등기술연구원\바탕 화면\BRepNet_share-master"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "src"))
from src.instance_common import (load_gt_instances, load_seg, load_face_attrs,
                                 group_by_connected_components, evaluate_part, aggregate_metrics)
DATA = r"D:\BRepNet_exp\dataset\MFInstSeg"
ATTRS = r"D:\BRepNet_exp\attrs"
NNCONV_SEG = r"D:\BRepNet_exp\runs\output_20260812_153230\test_output\seg"
STOCK = 24


def components(faces, neigh):
    faces = set(faces); seen = set(); n = 0
    for f in faces:
        if f in seen:
            continue
        n += 1; stack = [f]; seen.add(f)
        while stack:
            x = stack.pop()
            for y in neigh[x]:
                if y in faces and y not in seen:
                    seen.add(y); stack.append(y)
    return n


def main():
    stems = sorted(f[:-11] for f in os.listdir(ATTRS) if f.endswith(".attrs.json")
                   and os.path.exists(os.path.join(DATA, "inst", f[:-11] + ".inst.json")))
    mf30 = sorted(f[:-4] for f in os.listdir(NNCONV_SEG) if f.endswith(".seg"))[:30]
    per_part = {}
    tier_tot = Counter(); tier_30 = Counter()
    cls_touch = Counter()
    rows_gt_cc = []
    for s in stems:
        attrs = load_face_attrs(os.path.join(ATTRS, s + ".attrs.json"))
        neigh = defaultdict(set)
        for a, b in attrs["adjacency"]:
            neigh[a].add(b); neigh[b].add(a)
        gt = load_gt_instances(os.path.join(DATA, "inst", s + ".inst.json"))
        owner = {}
        for g, gi in enumerate(gt):
            for f in gi["faces"]:
                owner[f] = g
        tiers = []
        for g, gi in enumerate(gt):
            fs = set(gi["faces"])
            touching = any(owner.get(y) not in (None, g) and gt[owner[y]]["type"] == gi["type"]
                           for x in fs for y in neigh[x])
            disc = components(fs, neigh) > 1
            t = "C_disconnected" if disc else ("B_touching_same_class" if touching else "A_trivial")
            tiers.append(t)
            if touching:
                cls_touch[gi["type"]] += 1
        c = Counter(tiers)
        per_part[s] = {"n_inst": len(gt), "tiers": dict(c), "hard": (c["B_touching_same_class"] + c["C_disconnected"]) > 0}
        tier_tot.update(c)
        if s in mf30:
            tier_30.update(c)
        seg = load_seg(os.path.join(DATA, "seg", s + ".seg"))
        pred = group_by_connected_components(seg, attrs["adjacency"], (STOCK,))
        rows_gt_cc.append(evaluate_part(pred, gt, 0.5))
    hard = sorted(s for s, v in per_part.items() if v["hard"])
    hard30 = [s for s in mf30 if per_part[s]["hard"]]
    m = aggregate_metrics(rows_gt_cc)
    out = {"n_parts": len(stems), "n_instances": sum(tier_tot.values()), "tiers_all": dict(tier_tot),
           "n_hard_parts": len(hard), "hard_parts": hard,
           "mfinst30": {"n_instances": sum(tier_30.values()), "tiers": dict(tier_30), "n_hard_parts": len(hard30), "hard_parts": hard30},
           "rule_cc_with_gt_labels_500": {k: m[k] for k in ("precision", "recall", "f1", "n_parts")},
           "touching_by_class": dict(cls_touch), "per_part": per_part}
    with open(r"D:\BRepNet_exp\matrix\mfinstseg_hard_subset.json", "w", encoding="utf8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    n = out["n_instances"]
    print(f"parts={len(stems)} instances={n}")
    for k, v in sorted(tier_tot.items()):
        print(f"  {k:<24}{v:>6} ({100*v/n:.1f}%)")
    print(f"hard parts (>=1 tier B/C instance): {len(hard)} / {len(stems)}")
    n30 = out["mfinst30"]["n_instances"]
    print(f"mfinst30: instances={n30}", {k: f"{v} ({100*v/n30:.1f}%)" for k, v in sorted(tier_30.items())},
          f"hard parts {len(hard30)}/30")
    print("rule-CC with GT labels on 500:", {k: round(v, 4) if isinstance(v, float) else v for k, v in out["rule_cc_with_gt_labels_500"].items()})
    print("touching instances by class:", sorted(cls_touch.items(), key=lambda kv: -kv[1])[:8])


if __name__ == "__main__":
    main()

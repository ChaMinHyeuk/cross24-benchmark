# -*- coding: utf-8 -*-
"""리뷰 대응 분석 — 저장된 예측만으로 계산 (API 호출 없음).
(1) 전 방법 × IoU 임계값 0.5/0.75/0.9/1.0 F1
(2) cross24 oracle exclusive partition (공유 면을 한 인스턴스에만 배정했을 때 도달 가능한 최대 F1)
(3) 클래스 무관(class-agnostic) 매칭 F1 — 라벨 오류 분리
(4) mfinst30 미매칭 GT 인스턴스의 1단계 라벨 오류 귀속
출력: D:\\BRepNet_exp\\matrix\\reviewer_analysis.json + 콘솔
"""
import json, os, sys, itertools
from collections import Counter, defaultdict
REPO = r"c:\Users\차민혁\OneDrive - 고등기술연구원\바탕 화면\BRepNet_share-master"
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "src"))
from src.instance_common import (load_gt_instances, load_seg, load_face_attrs,
                                 group_by_connected_components, instance_iou)
DATA = r"D:\BRepNet_exp\dataset\MFInstSeg"
NNCONV_SEG = r"D:\BRepNet_exp\runs\output_20260812_153230\test_output\seg"
CROSS = r"D:\BRepNet_exp\cross_exp"
HELD = r"D:\BRepNet_exp\heldout_exp"
BENCH_DIR = {"cross24": CROSS, "heldout": HELD}
STOCK = 24
TAUS = (0.5, 0.75, 0.9, 1.0)


def stems(bench):
    if bench == "mfinst30":
        return sorted(f[:-4] for f in os.listdir(NNCONV_SEG) if f.endswith(".seg"))[:30]
    return sorted(f[:-10] for f in os.listdir(os.path.join(BENCH_DIR[bench], "inst")))


def gt_of(bench, stem):
    d = os.path.join(DATA, "inst") if bench == "mfinst30" else os.path.join(BENCH_DIR[bench], "inst")
    return load_gt_instances(os.path.join(d, stem + ".inst.json"))


def match(pred, gt, tau, class_aware=True):
    cands = []
    for p, pi in enumerate(pred):
        for g, gi in enumerate(gt):
            if class_aware and pi["type"] != gi["type"]:
                continue
            iou = instance_iou(pi["faces"], gi["faces"])
            if iou >= tau:
                cands.append((iou, p, g))
    cands.sort(reverse=True)
    up, ug, tp = set(), set(), 0
    for iou, p, g in cands:
        if p in up or g in ug:
            continue
        up.add(p); ug.add(g); tp += 1
    return tp, ug


def prf(tp, npred, ngt):
    P = tp / npred if npred else 0.0
    R = tp / ngt if ngt else 0.0
    return {"P": P, "R": R, "F1": 2 * P * R / (P + R) if P + R else 0.0,
            "tp": tp, "n_pred": npred, "n_gt": ngt}


def cc_pred(bench, stem, seg_dir):
    seg = load_seg(os.path.join(seg_dir, stem + ".seg"))
    attrs_dir = r"D:\BRepNet_exp\attrs" if bench == "mfinst30" else os.path.join(BENCH_DIR[bench], "attrs")
    attrs = load_face_attrs(os.path.join(attrs_dir, stem + ".attrs.json"))
    return group_by_connected_components(seg, attrs["adjacency"], (STOCK,))


def load_json_dir(d):
    return lambda s: json.load(open(os.path.join(d, s + ".inst.json"), encoding="UTF8"))


def methods(bench):
    m = {}
    seg_dir = NNCONV_SEG if bench == "mfinst30" else os.path.join(BENCH_DIR[bench], "seg")
    m["rule-CC"] = lambda s: cc_pred(bench, s, seg_dir)
    if bench == "mfinst30":
        m["rule-CC (GT labels)"] = lambda s: cc_pred(bench, s, os.path.join(DATA, "seg"))
    dirs = {"AAGNet-official": rf"D:\BRepNet_exp\aagnet_official\{bench}",
            "AAG-classic": rf"D:\BRepNet_exp\inst_head\aagclassic_{bench}",
            "edge-cut": rf"D:\BRepNet_exp\inst_head\edgecut_{bench}",
            "disc-embedding": rf"D:\BRepNet_exp\inst_head\disc_{bench}"}
    for slug in sorted(os.listdir(r"D:\BRepNet_exp\matrix")):
        d = rf"D:\BRepNet_exp\matrix\{slug}\{bench}"
        if os.path.isdir(d):
            dirs["LLM:" + slug] = d
    for name, d in dirs.items():
        # 실행 중(부분 출력)인 디렉터리는 제외 — 전 파트가 있어야 집계
        if os.path.isdir(d) and all(os.path.exists(os.path.join(d, s + ".inst.json")) for s in stems(bench)):
            m[name] = load_json_dir(d)
    return m


def variant(stem):
    return stem.rsplit("_", 1)[0]


def oracle_exclusive(gt):
    """공유 면 각각을 소유 인스턴스 중 하나에만 배정하는 모든 조합 중 τ별 최대 tp."""
    owners = defaultdict(list)
    for g, gi in enumerate(gt):
        for f in gi["faces"]:
            owners[f].append(g)
    shared = [f for f, o in owners.items() if len(o) > 1]
    best = {t: 0 for t in TAUS}
    for choice in itertools.product(*[owners[f] for f in shared]):
        keep = dict(zip(shared, choice))
        pred = [{"type": gi["type"],
                 "faces": [f for f in gi["faces"] if f not in keep or keep[f] == g]}
                for g, gi in enumerate(gt)]
        for t in TAUS:
            best[t] = max(best[t], match(pred, gt, t)[0])
    return best, len(shared)


def score(preds, gts, ss, t, class_aware=True):
    tp = sum(match(preds[s], gts[s], t, class_aware)[0] for s in ss)
    return prf(tp, sum(len(preds[s]) for s in ss), sum(len(gts[s]) for s in ss))


def main():
    out = {}
    for bench in ("mfinst30", "cross24", "heldout"):
        S = stems(bench)
        gts = {s: gt_of(bench, s) for s in S}
        out[bench] = {}
        for name, fn in methods(bench).items():
            preds = {s: fn(s) for s in S}
            row = {f"F1@{t}": score(preds, gts, S, t) for t in TAUS}
            row["F1@0.5 class-agnostic"] = score(preds, gts, S, 0.5, False)
            if bench != "mfinst30":
                row["per_variant"] = {}
                for v in sorted({variant(s) for s in S}):
                    ss = [s for s in S if variant(s) == v]
                    pv = {f"F1@{t}": round(score(preds, gts, ss, t)["F1"], 3) for t in TAUS}
                    pv["n_pred/n_gt"] = f"{sum(len(preds[s]) for s in ss)}/{sum(len(gts[s]) for s in ss)}"
                    row["per_variant"][v] = pv
            out[bench][name] = row

        if bench != "mfinst30":
            tot = {t: 0 for t in TAUS}
            ngt = 0
            nshared = 0
            pv = defaultdict(lambda: {t: [0, 0] for t in TAUS})
            for s in S:
                b, k = oracle_exclusive(gts[s])
                nshared += k
                ngt += len(gts[s])
                for t in TAUS:
                    tot[t] += b[t]
                    pv[variant(s)][t][0] += b[t]
                    pv[variant(s)][t][1] += len(gts[s])
            row = {f"F1@{t}": prf(tot[t], ngt, ngt) for t in TAUS}
            row["F1@0.5 class-agnostic"] = row["F1@0.5"]
            row["per_variant"] = {v: {f"F1@{t}": round(prf(a, n, n)["F1"], 3) for t, (a, n) in d.items()}
                                  for v, d in pv.items()}
            row["n_shared_faces_total"] = nshared
            out[bench]["ORACLE exclusive partition"] = row
        else:
            gtseg = {s: load_seg(os.path.join(DATA, "seg", s + ".seg")) for s in S}
            nnseg = {s: load_seg(os.path.join(NNCONV_SEG, s + ".seg")) for s in S}
            n_face = sum(len(gtseg[s]) for s in S)
            face_acc = sum(a == b for s in S for a, b in zip(gtseg[s], nnseg[s])) / n_face
            out[bench]["_nnconv_face_acc_on_mfinst30"] = face_acc
            for name in ("LLM:gpt-5.5", "LLM:gemini-3.1-pro", "AAGNet-official", "rule-CC"):
                fn = methods(bench)[name]
                attr = Counter()
                for s in S:
                    pred = fn(s)
                    gt = gts[s]
                    _, matched = match(pred, gt, 0.5)
                    _, m2 = match(pred, gt, 0.5, class_aware=False)
                    for g, gi in enumerate(gt):
                        if g in matched:
                            continue
                        maj = Counter(nnseg[s][f] for f in gi["faces"]).most_common(1)[0][0]
                        wrong_any = any(nnseg[s][f] != gi["type"] for f in gi["faces"])
                        attr["unmatched_total"] += 1
                        if maj != gi["type"]:
                            attr["stage1_majority_label_wrong"] += 1
                        elif wrong_any:
                            attr["stage1_some_face_wrong"] += 1
                        else:
                            attr["stage1_labels_all_correct"] += 1
                        if g in m2:
                            attr["recovered_by_class_agnostic"] += 1
                out[bench][name]["unmatched_gt_attribution"] = dict(attr)

    with open(r"D:\BRepNet_exp\matrix\reviewer_analysis.json", "w", encoding="utf8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    for bench in out:
        print(f"\n=== {bench} ===")
        print(f"{'method':<28}" + "".join(f"{'F1@' + str(t):>9}" for t in TAUS) + f"{'cls-agn':>9}{'n_pred/gt':>11}")
        for name, row in out[bench].items():
            if name.startswith("_"):
                print(name, round(row, 4))
                continue
            line = f"{name:<28}" + "".join(f"{row[f'F1@{t}']['F1']:>9.3f}" for t in TAUS)
            line += f"{row['F1@0.5 class-agnostic']['F1']:>9.3f}"
            line += f"{row['F1@0.5']['n_pred']:>6}/{row['F1@0.5']['n_gt']}"
            print(line)
            if "unmatched_gt_attribution" in row:
                print("      unmatched GT attribution:", row["unmatched_gt_attribution"])
        if bench != "mfinst30":
            print("\n-- per variant F1 @ 0.5 / 0.75 / 0.9 / 1.0  (n_pred/n_gt)")
            for name, row in out[bench].items():
                if "per_variant" not in row:
                    continue
                cells = []
                for v, d in row["per_variant"].items():
                    cells.append(f"{v}: " + "/".join(f"{d[f'F1@{t}']:.2f}" for t in TAUS)
                                 + (f" ({d['n_pred/n_gt']})" if "n_pred/n_gt" in d else ""))
                print(f"{name:<28} " + " | ".join(cells))


if __name__ == "__main__":
    main()

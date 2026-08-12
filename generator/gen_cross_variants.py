# -*- coding: utf-8 -*-
"""교차 패턴 다양화 생성기 — 프롬프트에 예시가 없는 유형들.

variant A) cross_diff  : 깊이가 다른 직교 관통 슬롯 교차
   - 얕은 슬롯 A의 바닥이 깊은 슬롯 B에 의해 2조각으로 분할 (면 공유 없음, 맞물림)
   - GT: A = 벽4 + 바닥2 (6면), B = 벽2(L자 병합) + 바닥1 (3면), 모두 class 6
variant B) slot_pocket : 같은 깊이의 슬롯 × 포켓 이종 교차
   - 병합된 바닥면 1개를 포켓(14)과 슬롯(6) 인스턴스가 공유
   - GT: 포켓 = y벽2 + x벽4 + 공유바닥 (7면), 슬롯 = 벽4 + 공유바닥 (5면)

출력: D:\\BRepNet_exp\\cross_exp\\{steps,seg,inst}\\<variant>_NNN.*
사용: python gen_cross_variants.py <n_per_variant>
"""
import json
import os
import random
import sys

sys.path.insert(0, r"D:\BRepNet_exp\shims")

from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import GeomAbs_Plane
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.gp import gp_Pnt
from OCC.Core.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCC.Extend.DataExchange import write_step_file

from occwl.compound import Compound
from occwl.graph import face_adjacency

OUT = r"D:\BRepNet_exp\cross_exp"
SLOT = 6      # rectangular_through_slot
POCKET = 14   # rectangular_pocket
STOCK = 24
TOL = 1e-3
EPS = 1.0


def box(x1, y1, z1, x2, y2, z2):
    return BRepPrimAPI_MakeBox(gp_Pnt(x1, y1, z1), gp_Pnt(x2, y2, z2)).Shape()


def cut_and_unify(stock, tools):
    fused = tools[0]
    for t in tools[1:]:
        fused = BRepAlgoAPI_Fuse(fused, t).Shape()
    cut = BRepAlgoAPI_Cut(stock, fused).Shape()
    unify = ShapeUpgrade_UnifySameDomain(cut, True, True, False)
    unify.Build()
    return unify.Shape()


def plane_info(face):
    td = face.topods_shape() if hasattr(face, "topods_shape") else face.topods_face()
    adaptor = BRepAdaptor_Surface(td)
    if adaptor.GetType() != GeomAbs_Plane:
        return None
    n = adaptor.Plane().Axis().Direction()
    props = GProp_GProps()
    brepgprop.SurfaceProperties(td, props)
    c = props.CentreOfMass()
    return (abs(n.X()), abs(n.Y()), abs(n.Z())), (c.X(), c.Y(), c.Z())


def classify(step_path, rules):
    """rules: [(역할이름, 판정함수(n, c))] — 첫 매칭 규칙의 역할 부여, 없으면 stock."""
    solid = Compound.load_from_step(step_path)
    graph = face_adjacency(solid)
    roles = {}
    for idx in graph.nodes:
        info = plane_info(graph.nodes[idx]["face"])
        roles[idx] = "stock"
        if info is None:
            continue
        n, c = info
        for name, fn in rules:
            if fn(n, c):
                roles[idx] = name
                break
    return roles, len(graph.nodes)


def near(v, t):
    return abs(v - t) < TOL


def gen_cross_diff(rng, idx):
    """깊이 다른 직교 관통 슬롯."""
    L, W, H = rng.uniform(60, 100), rng.uniform(60, 100), rng.uniform(28, 40)
    wa, wb = rng.uniform(8, 16), rng.uniform(8, 16)
    dA = rng.uniform(6, H / 2 - 3)
    dB = rng.uniform(dA + 5, H - 6)
    y0 = rng.uniform(0.2 * W, 0.8 * W - wa)
    x0 = rng.uniform(0.2 * L, 0.8 * L - wb)

    shape = cut_and_unify(
        box(0, 0, 0, L, W, H),
        [box(-EPS, y0, H - dA, L + EPS, y0 + wa, H + EPS),     # A: 얕은, x방향
         box(x0, -EPS, H - dB, x0 + wb, W + EPS, H + EPS)])    # B: 깊은, y방향

    stem = f"cross_diff_{idx:03d}"
    step_path = os.path.join(OUT, "steps", stem + ".step")
    write_step_file(shape, step_path)

    rules = [
        ("bottomA", lambda n, c: n[2] > 0.99 and near(c[2], H - dA)),
        ("bottomB", lambda n, c: n[2] > 0.99 and near(c[2], H - dB)),
        ("wallA", lambda n, c: n[1] > 0.99 and (near(c[1], y0) or near(c[1], y0 + wa)) and c[2] > H - dA - TOL),
        ("wallB", lambda n, c: n[0] > 0.99 and (near(c[0], x0) or near(c[0], x0 + wb)) and c[2] > H - dB - TOL),
    ]
    roles, n_faces = classify(step_path, rules)
    counts = {r: sum(1 for v in roles.values() if v == r) for r in ("bottomA", "bottomB", "wallA", "wallB")}
    if not (counts["bottomA"] == 2 and counts["bottomB"] == 1 and counts["wallA"] == 4 and counts["wallB"] == 2):
        os.remove(step_path)
        return None, f"{stem}: unexpected topology {counts} (faces={n_faces})"

    seg = [SLOT if roles[i] != "stock" else STOCK for i in range(n_faces)]
    inst = [
        {"type": SLOT, "faces": sorted(i for i, r in roles.items() if r in ("wallA", "bottomA"))},
        {"type": SLOT, "faces": sorted(i for i, r in roles.items() if r in ("wallB", "bottomB"))},
    ]
    return (stem, seg, inst), None


def gen_slot_pocket(rng, idx):
    """같은 깊이 슬롯 × 포켓 (이종 교차, 병합 바닥 공유)."""
    L, W, H = rng.uniform(70, 110), rng.uniform(70, 110), rng.uniform(24, 36)
    d = rng.uniform(6, H - 8)
    pl, pw = rng.uniform(24, 40), rng.uniform(24, 40)          # 포켓 크기
    px0 = rng.uniform(0.15 * L, 0.85 * L - pl)
    py0 = rng.uniform(0.15 * W, 0.85 * W - pw)
    sw = rng.uniform(7, pw - 8)                                 # 슬롯 폭 < 포켓 폭
    sy0 = rng.uniform(py0 + 3, py0 + pw - sw - 3)               # 슬롯이 포켓 y범위 안쪽 통과

    shape = cut_and_unify(
        box(0, 0, 0, L, W, H),
        [box(px0, py0, H - d, px0 + pl, py0 + pw, H + EPS),     # 포켓 (blind)
         box(-EPS, sy0, H - d, L + EPS, sy0 + sw, H + EPS)])    # 슬롯 (x관통, 같은 깊이)

    stem = f"slot_pocket_{idx:03d}"
    step_path = os.path.join(OUT, "steps", stem + ".step")
    write_step_file(shape, step_path)

    rules = [
        ("bottom", lambda n, c: n[2] > 0.99 and near(c[2], H - d)),
        ("pWallY", lambda n, c: n[1] > 0.99 and (near(c[1], py0) or near(c[1], py0 + pw)) and c[2] > H - d - TOL),
        ("sWall", lambda n, c: n[1] > 0.99 and (near(c[1], sy0) or near(c[1], sy0 + sw)) and c[2] > H - d - TOL),
        ("pWallX", lambda n, c: n[0] > 0.99 and (near(c[0], px0) or near(c[0], px0 + pl)) and c[2] > H - d - TOL),
    ]
    roles, n_faces = classify(step_path, rules)
    counts = {r: sum(1 for v in roles.values() if v == r) for r in ("bottom", "pWallY", "sWall", "pWallX")}
    if not (counts["bottom"] == 1 and counts["pWallY"] == 2 and counts["sWall"] == 4 and counts["pWallX"] == 4):
        os.remove(step_path)
        return None, f"{stem}: unexpected topology {counts} (faces={n_faces})"

    seg = [STOCK] * n_faces
    for i, r in roles.items():
        if r in ("pWallY", "pWallX"):
            seg[i] = POCKET
        elif r == "sWall":
            seg[i] = SLOT
        elif r == "bottom":
            seg[i] = SLOT  # 병합 바닥의 면 단위 라벨은 하나만 가능 — 슬롯으로 배정
    bottom = [i for i, r in roles.items() if r == "bottom"]
    inst = [
        {"type": POCKET, "faces": sorted([i for i, r in roles.items() if r in ("pWallY", "pWallX")] + bottom)},
        {"type": SLOT, "faces": sorted([i for i, r in roles.items() if r == "sWall"] + bottom)},
    ]
    return (stem, seg, inst), None


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    rng = random.Random(7)
    for sub in ("steps", "seg", "inst"):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)

    for gen, label in ((gen_cross_diff, "cross_diff"), (gen_slot_pocket, "slot_pocket")):
        made, tries = 0, 0
        while made < n and tries < n * 4:
            result, err = gen(rng, made)
            tries += 1
            if err:
                print("skip", err)
                continue
            stem, seg, inst = result
            with open(os.path.join(OUT, "seg", stem + ".seg"), "w") as f:
                f.write("\n".join(map(str, seg)) + "\n")
            with open(os.path.join(OUT, "inst", stem + ".inst.json"), "w") as f:
                json.dump(inst, f)
            made += 1
            print(f"ok {stem}: {len(seg)} faces, inst sizes {[len(i['faces']) for i in inst]}")
        print(f"{label}: {made}/{n}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Held-out 교차 변형 생성기 — 프롬프트 v2 동결 이후에 설계된 유형 (프롬프트에 예시 없음).

variant H1) triple_cross : 같은 깊이 관통 슬롯 3개(x방향 2개 + y방향 1개) — 병합 바닥 1면을 3개 인스턴스가 공유
   - GT: A1 = 벽4(y방향 슬롯에 의해 2조각씩) + 공유바닥, A2 동일, B = 벽6(3조각씩) + 공유바닥, 모두 class 6
variant H2) oblique_cross: 같은 깊이 관통 슬롯 2개가 비직교(30~60°)로 교차 — 병합 바닥 1면 공유
   - GT: A = 벽4 + 공유바닥, B(경사) = 벽4 + 공유바닥, 모두 class 6
variant H3) hole_slot    : 관통 슬롯(class 6)을 지름이 슬롯 폭보다 큰 수직 관통 홀(class 1)이 절단
   - GT: 슬롯 = 벽4 + 바닥2조각(단절), 홀 = 원통면 전부 (면 공유 없음, 곡면 경계 단절)

출력: D:\\BRepNet_exp\\heldout_exp\\{steps,seg,inst}\\<variant>_NNN.*
사용: python gen_heldout_variants.py <n_per_variant>   (시드 11 고정)
"""
import json
import math
import os
import random
import sys

sys.path.insert(0, r"D:\BRepNet_exp\shims")

from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax1, gp_Ax2, gp_Trsf
from OCC.Core.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCC.Extend.DataExchange import write_step_file

from occwl.compound import Compound
from occwl.graph import face_adjacency

OUT = r"D:\BRepNet_exp\heldout_exp"
SLOT = 6      # rectangular_through_slot
HOLE = 1      # through_hole
STOCK = 24
TOL = 1e-3
EPS = 1.0


def box(x1, y1, z1, x2, y2, z2):
    return BRepPrimAPI_MakeBox(gp_Pnt(x1, y1, z1), gp_Pnt(x2, y2, z2)).Shape()


def rotated_box(cx, cy, z1, z2, length, width, theta):
    """(cx,cy)를 중심으로 z축 기준 theta(rad) 회전한 length×width 박스 (z1..z2)."""
    b = box(cx - length / 2, cy - width / 2, z1, cx + length / 2, cy + width / 2, z2)
    tr = gp_Trsf()
    tr.SetRotation(gp_Ax1(gp_Pnt(cx, cy, 0), gp_Dir(0, 0, 1)), theta)
    return BRepBuilderAPI_Transform(b, tr, True).Shape()


def cylinder(cx, cy, z1, z2, r):
    return BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(cx, cy, z1), gp_Dir(0, 0, 1)), r, z2 - z1).Shape()


def cut_and_unify(stock, tools):
    fused = tools[0]
    for t in tools[1:]:
        fused = BRepAlgoAPI_Fuse(fused, t).Shape()
    cut = BRepAlgoAPI_Cut(stock, fused).Shape()
    unify = ShapeUpgrade_UnifySameDomain(cut, True, True, False)
    unify.Build()
    return unify.Shape()


def surf_info(face):
    td = face.topods_shape() if hasattr(face, "topods_shape") else face.topods_face()
    adaptor = BRepAdaptor_Surface(td)
    props = GProp_GProps()
    brepgprop.SurfaceProperties(td, props)
    c = props.CentreOfMass()
    c = (c.X(), c.Y(), c.Z())
    t = adaptor.GetType()
    if t == GeomAbs_Plane:
        n = adaptor.Plane().Axis().Direction()
        return "plane", (abs(n.X()), abs(n.Y()), abs(n.Z())), c
    if t == GeomAbs_Cylinder:
        return "cyl", None, c
    return "other", None, c


def classify(step_path, rules):
    """rules: [(역할, 판정함수(kind, n, c))] — 첫 매칭 규칙의 역할, 없으면 stock."""
    solid = Compound.load_from_step(step_path)
    graph = face_adjacency(solid)
    roles = {}
    for idx in graph.nodes:
        kind, n, c = surf_info(graph.nodes[idx]["face"])
        roles[idx] = "stock"
        for name, fn in rules:
            if fn(kind, n, c):
                roles[idx] = name
                break
    return roles, len(graph.nodes)


def near(v, t):
    return abs(v - t) < TOL


def counts_of(roles, names):
    return {r: sum(1 for v in roles.values() if v == r) for r in names}


def faces_of(roles, *names):
    return sorted(i for i, r in roles.items() if r in names)


def write_step(shape, stem):
    p = os.path.join(OUT, "steps", stem + ".step")
    write_step_file(shape, p)
    return p


# ---------------------------------------------------------------------------
def gen_triple_cross(rng, idx):
    L, W, H = rng.uniform(80, 120), rng.uniform(80, 120), rng.uniform(24, 36)
    d = rng.uniform(6, H - 8)
    w1, w2, wb = rng.uniform(8, 14), rng.uniform(8, 14), rng.uniform(8, 14)
    a1 = rng.uniform(0.12 * W, 0.40 * W - w1)
    a2 = rng.uniform(0.60 * W, 0.88 * W - w2)
    b = rng.uniform(0.25 * L, 0.75 * L - wb)
    shape = cut_and_unify(box(0, 0, 0, L, W, H), [
        box(-EPS, a1, H - d, L + EPS, a1 + w1, H + EPS),
        box(-EPS, a2, H - d, L + EPS, a2 + w2, H + EPS),
        box(b, -EPS, H - d, b + wb, W + EPS, H + EPS)])
    stem = f"triple_cross_{idx:03d}"
    step_path = write_step(shape, stem)
    above = lambda c: c[2] > H - d - TOL
    rules = [
        ("bottom", lambda k, n, c: k == "plane" and n[2] > 0.99 and near(c[2], H - d)),
        ("wallA1", lambda k, n, c: k == "plane" and n[1] > 0.99 and (near(c[1], a1) or near(c[1], a1 + w1)) and above(c)),
        ("wallA2", lambda k, n, c: k == "plane" and n[1] > 0.99 and (near(c[1], a2) or near(c[1], a2 + w2)) and above(c)),
        ("wallB", lambda k, n, c: k == "plane" and n[0] > 0.99 and (near(c[0], b) or near(c[0], b + wb)) and above(c)),
    ]
    roles, n_faces = classify(step_path, rules)
    cnt = counts_of(roles, ("bottom", "wallA1", "wallA2", "wallB"))
    if not (cnt["bottom"] == 1 and cnt["wallA1"] == 4 and cnt["wallA2"] == 4 and cnt["wallB"] == 6):
        os.remove(step_path)
        return None, f"{stem}: unexpected topology {cnt} (faces={n_faces})"
    seg = [SLOT if roles[i] != "stock" else STOCK for i in range(n_faces)]
    bottom = faces_of(roles, "bottom")
    inst = [{"type": SLOT, "faces": sorted(faces_of(roles, "wallA1") + bottom)},
            {"type": SLOT, "faces": sorted(faces_of(roles, "wallA2") + bottom)},
            {"type": SLOT, "faces": sorted(faces_of(roles, "wallB") + bottom)}]
    return (stem, seg, inst), None


def gen_oblique_cross(rng, idx):
    L, W, H = rng.uniform(80, 120), rng.uniform(80, 120), rng.uniform(24, 36)
    d = rng.uniform(6, H - 8)
    wa, wb = rng.uniform(8, 14), rng.uniform(8, 14)
    y0 = rng.uniform(0.35 * W, 0.65 * W - wa)
    theta = math.radians(rng.uniform(30, 60)) * rng.choice((1, -1))
    cx, cy = rng.uniform(0.35 * L, 0.65 * L), y0 + wa / 2
    shape = cut_and_unify(box(0, 0, 0, L, W, H), [
        box(-EPS, y0, H - d, L + EPS, y0 + wa, H + EPS),
        rotated_box(cx, cy, H - d, H + EPS, 4 * max(L, W), wb, theta)])
    stem = f"oblique_cross_{idx:03d}"
    step_path = write_step(shape, stem)
    above = lambda c: c[2] > H - d - TOL
    sn, cn = abs(math.sin(theta)), abs(math.cos(theta))
    rules = [
        ("bottom", lambda k, n, c: k == "plane" and n[2] > 0.99 and near(c[2], H - d)),
        ("wallA", lambda k, n, c: k == "plane" and n[1] > 0.99 and (near(c[1], y0) or near(c[1], y0 + wa)) and above(c)),
        ("wallB", lambda k, n, c: k == "plane" and abs(n[0] - sn) < 1e-3 and abs(n[1] - cn) < 1e-3 and n[2] < 1e-3 and above(c)),
    ]
    roles, n_faces = classify(step_path, rules)
    cnt = counts_of(roles, ("bottom", "wallA", "wallB"))
    if not (cnt["bottom"] == 1 and cnt["wallA"] == 4 and cnt["wallB"] == 4):
        os.remove(step_path)
        return None, f"{stem}: unexpected topology {cnt} (faces={n_faces})"
    seg = [SLOT if roles[i] != "stock" else STOCK for i in range(n_faces)]
    bottom = faces_of(roles, "bottom")
    inst = [{"type": SLOT, "faces": sorted(faces_of(roles, "wallA") + bottom)},
            {"type": SLOT, "faces": sorted(faces_of(roles, "wallB") + bottom)}]
    return (stem, seg, inst), None


def gen_hole_slot(rng, idx):
    L, W, H = rng.uniform(80, 120), rng.uniform(80, 120), rng.uniform(24, 36)
    d = rng.uniform(6, H - 8)
    w = rng.uniform(8, 14)
    y0 = rng.uniform(0.3 * W, 0.7 * W - w)
    r = rng.uniform(0.65 * w, 0.95 * w)             # 지름 > 슬롯 폭 → 바닥·벽 절단
    cx, cy = rng.uniform(0.3 * L, 0.7 * L), y0 + w / 2
    shape = cut_and_unify(box(0, 0, 0, L, W, H), [
        box(-EPS, y0, H - d, L + EPS, y0 + w, H + EPS),
        cylinder(cx, cy, -EPS, H + EPS, r)])
    stem = f"hole_slot_{idx:03d}"
    step_path = write_step(shape, stem)
    above = lambda c: c[2] > H - d - TOL
    rules = [
        ("cyl", lambda k, n, c: k == "cyl"),
        ("bottom", lambda k, n, c: k == "plane" and n[2] > 0.99 and near(c[2], H - d)),
        ("wall", lambda k, n, c: k == "plane" and n[1] > 0.99 and (near(c[1], y0) or near(c[1], y0 + w)) and above(c)),
    ]
    roles, n_faces = classify(step_path, rules)
    cnt = counts_of(roles, ("cyl", "bottom", "wall"))
    if not (cnt["bottom"] == 2 and cnt["wall"] == 4 and cnt["cyl"] >= 1):
        os.remove(step_path)
        return None, f"{stem}: unexpected topology {cnt} (faces={n_faces})"
    seg = [STOCK] * n_faces
    for i, rr in roles.items():
        if rr in ("bottom", "wall"):
            seg[i] = SLOT
        elif rr == "cyl":
            seg[i] = HOLE
    inst = [{"type": SLOT, "faces": faces_of(roles, "wall", "bottom")},
            {"type": HOLE, "faces": faces_of(roles, "cyl")}]
    return (stem, seg, inst), None


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    rng = random.Random(11)
    for sub in ("steps", "seg", "inst"):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)
    report = {}
    for gen, label in ((gen_triple_cross, "triple_cross"), (gen_oblique_cross, "oblique_cross"),
                       (gen_hole_slot, "hole_slot")):
        made, tries, rejects = 0, 0, []
        while made < n and tries < n * 4:
            result, err = gen(rng, made)
            tries += 1
            if err:
                print("skip", err)
                rejects.append(err)
                continue
            stem, seg, inst = result
            with open(os.path.join(OUT, "seg", stem + ".seg"), "w") as f:
                f.write("\n".join(map(str, seg)) + "\n")
            with open(os.path.join(OUT, "inst", stem + ".inst.json"), "w") as f:
                json.dump(inst, f)
            made += 1
            print(f"ok {stem}: {len(seg)} faces, inst sizes {[len(i['faces']) for i in inst]}")
        report[label] = {"made": made, "tries": tries, "rejects": rejects}
        print(f"{label}: {made}/{n} (tries {tries})")
    with open(os.path.join(OUT, "generation_report.json"), "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()

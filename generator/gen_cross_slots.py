"""십자 교차 슬롯 합성 파트 생성기 — 면 공유 GT 검증용.

스톡 박스에 직교하는 관통 슬롯 2개(같은 깊이)를 십자로 커팅:
- 교차부 바닥은 하나의 병합된 십자형 평면 → 두 슬롯 인스턴스가 이 면을 공유
- GT .inst.json은 공유 바닥면을 두 인스턴스 모두에 배정 (면 공유 표현)
- .seg: 슬롯 면 = rectangular_through_slot(6), 나머지 = stock(24)

출력: D:\BRepNet_exp\cross_exp\{steps,seg,inst}
면 id는 occwl face_adjacency 노드 순서 (파이프라인 전체와 동일).
사용: python gen_cross_slots.py <n_parts>
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
SLOT_CLASS = 6    # rectangular_through_slot
STOCK_CLASS = 24
TOL = 1e-3
EPS = 1.0         # 관통 커팅용 여유


def box(x1, y1, z1, x2, y2, z2):
    return BRepPrimAPI_MakeBox(gp_Pnt(x1, y1, z1), gp_Pnt(x2, y2, z2)).Shape()


def make_part(L, W, H, wa, wb, d, y0, x0):
    """스톡 - (슬롯A ∪ 슬롯B). A: x방향 관통(y0~y0+wa), B: y방향 관통(x0~x0+wb), 깊이 d."""
    stock = box(0, 0, 0, L, W, H)
    slot_a = box(-EPS, y0, H - d, L + EPS, y0 + wa, H + EPS)
    slot_b = box(x0, -EPS, H - d, x0 + wb, W + EPS, H + EPS)
    cross = BRepAlgoAPI_Fuse(slot_a, slot_b).Shape()
    cut = BRepAlgoAPI_Cut(stock, cross).Shape()
    # Fuse가 남긴 내부 경계 제거 — 교차부 바닥을 하나의 십자형 면으로 병합
    unify = ShapeUpgrade_UnifySameDomain(cut, True, True, False)
    unify.Build()
    return unify.Shape()


def classify_faces(step_path, L, W, H, wa, wb, d, y0, x0):
    """면 id(occwl 순서) → 'bottom' | 'wallA' | 'wallB' | 'stock' 분류."""
    solid = Compound.load_from_step(step_path)
    graph = face_adjacency(solid)
    roles = {}
    for idx in graph.nodes:
        face = graph.nodes[idx]["face"]
        topods = face.topods_shape() if hasattr(face, "topods_shape") else face.topods_face()
        adaptor = BRepAdaptor_Surface(topods)
        if adaptor.GetType() != GeomAbs_Plane:
            roles[idx] = "stock"
            continue
        n = adaptor.Plane().Axis().Direction()
        props = GProp_GProps()
        brepgprop.SurfaceProperties(topods, props)
        c = props.CentreOfMass()
        cx, cy, cz = c.X(), c.Y(), c.Z()
        nx, ny, nz = abs(n.X()), abs(n.Y()), abs(n.Z())

        if nz > 0.99 and abs(cz - (H - d)) < TOL:
            roles[idx] = "bottom"          # 십자형 병합 바닥 (공유 면)
        elif ny > 0.99 and (abs(cy - y0) < TOL or abs(cy - (y0 + wa)) < TOL) and cz > H - d - TOL:
            roles[idx] = "wallA"
        elif nx > 0.99 and (abs(cx - x0) < TOL or abs(cx - (x0 + wb)) < TOL) and cz > H - d - TOL:
            roles[idx] = "wallB"
        else:
            roles[idx] = "stock"
    n_edges = len(graph.edges)
    return roles, len(graph.nodes), n_edges


def main():
    n_parts = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    rng = random.Random(42)
    for sub in ("steps", "seg", "inst"):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)

    made = 0
    i = 0
    while made < n_parts and i < n_parts * 3:
        i += 1
        L = rng.uniform(60, 100); W = rng.uniform(60, 100); H = rng.uniform(24, 40)
        wa = rng.uniform(8, 18); wb = rng.uniform(8, 18)
        d = rng.uniform(6, H - 8)
        y0 = rng.uniform(0.18 * W, 0.82 * W - wa)
        x0 = rng.uniform(0.18 * L, 0.82 * L - wb)
        stem = f"cross_slot_{made:03d}"
        step_path = os.path.join(OUT, "steps", stem + ".step")
        try:
            shape = make_part(L, W, H, wa, wb, d, y0, x0)
            write_step_file(shape, step_path)
            roles, n_faces, n_edges = classify_faces(step_path, L, W, H, wa, wb, d, y0, x0)

            n_bottom = sum(1 for r in roles.values() if r == "bottom")
            n_wa = sum(1 for r in roles.values() if r == "wallA")
            n_wb = sum(1 for r in roles.values() if r == "wallB")
            # 기대 토폴로지: 십자 바닥 1, 슬롯별 벽 4 (교차로 2분할된 2쌍)
            if not (n_bottom == 1 and n_wa == 4 and n_wb == 4):
                print(f"skip {stem}: bottom={n_bottom} wallA={n_wa} wallB={n_wb} faces={n_faces}")
                os.remove(step_path)
                continue

            seg = [SLOT_CLASS if roles[i] != "stock" else STOCK_CLASS for i in range(n_faces)]
            bottom_id = [i for i, r in roles.items() if r == "bottom"]
            inst = [
                {"type": SLOT_CLASS, "faces": sorted([i for i, r in roles.items() if r == "wallA"] + bottom_id)},
                {"type": SLOT_CLASS, "faces": sorted([i for i, r in roles.items() if r == "wallB"] + bottom_id)},
            ]
            with open(os.path.join(OUT, "seg", stem + ".seg"), "w") as f:
                f.write("\n".join(map(str, seg)) + "\n")
            with open(os.path.join(OUT, "inst", stem + ".inst.json"), "w") as f:
                json.dump(inst, f)
            made += 1
            print(f"ok {stem}: faces={n_faces} edges={n_edges} shared_bottom=face_{bottom_id[0]}")
        except Exception as e:
            print(f"fail {stem}: {e}")
            if os.path.exists(step_path):
                os.remove(step_path)

    print(f"generated {made} parts -> {OUT}")


if __name__ == "__main__":
    main()

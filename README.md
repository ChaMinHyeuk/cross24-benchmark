# cross24: A Synthetic Intersecting-Feature Benchmark for Instance-Level Machining Feature Recognition

This repository accompanies the paper *"Instance-Level Machining Feature Recognition via
Training-Free LLM Grouping"* (under review). It contains the **cross24** benchmark — to our
knowledge the first quantitative benchmark whose ground truth expresses **shared (overlapping)
face assignments** for intersecting machining features — together with the parametric generator,
the frozen LLM prompts, the raw LLM responses of all evaluated models, and the evaluation code.

## Why this benchmark exists

When two machining features intersect (e.g., two slots of equal depth crossing in a "+" shape),
the boundary of the merged removal volume contains faces that belong to **both** features at
once: the cross-shaped floor is simultaneously the floor of slot A and slot B. Any grouping
method that outputs a *partition* of faces (connected components, instance-adjacency matrix
clustering, instance-index classification) cannot even represent this ground truth. Public
datasets share the blind spot: MFInstSeg encodes instances as a same-instance adjacency matrix
whose connected components are disjoint by construction. cross24 fills this evaluation gap.

## Benchmark structure (24 parts, 3 variants)

| Variant | Parts | Construction | Ground-truth structure |
|---|---|---|---|
| `cross_slot_*` | 12 | two orthogonal through slots, equal depth | one cross-shaped merged floor **shared by both** slot instances |
| `cross_diff_*` | 6 | orthogonal through slots, different depths | no shared face; shallow slot floor split into two pieces (interlocked) |
| `slot_pocket_*` | 6 | through slot crossing a wider pocket, equal depth | merged floor **shared across heterogeneous types** (pocket + slot) |

Dimensions and placements are randomized with a fixed seed; the generated topology is verified
programmatically (expected wall/floor face counts; zero rejects in the released set).

```
benchmark/
  steps/            24 STEP (AP203) B-rep models
  gt_instances/     <part>.inst.json — [{"type": class_id, "faces": [face_id, ...]}, ...]
                    face ids may appear in MULTIPLE instances (shared faces)
  face_labels/      <part>.seg — per-face semantic class (one id per line, face order below)
  face_attributes/  <part>.attrs.json — per-face type/area/centroid/axis/radius +
                    adjacency with edge-convexity annotations (concave/convex/smooth)
generator/          parametric generators (pythonOCC/occwl over Open CASCADE)
prompts/            frozen system prompts v1 (no interrupted-instance rule) and v2 (final)
llm_responses/      raw responses + parsed predictions of all 12 evaluated LLMs
                    (per model, per benchmark), plus the prompt-v1 ablation run
evaluation/         evaluation protocol (greedy IoU>=0.5 matching with class agreement,
                    micro P/R/F1, mean matched IoU; supports overlapping instances)
```

**Face ordering.** Face ids follow the node order of the occwl `face_adjacency` graph of the
STEP model — the same convention used by the MFInstSeg/AAGNet extraction pipeline — so STEP
files, labels, attributes, and ground truth all refer to the same faces without any mapping.

**Class ids** follow the MFInstSeg taxonomy (0 = chamfer … 23 = round, 24 = stock);
`rectangular_through_slot` = 6, `rectangular_pocket` = 14.

## Evaluation protocol

Predicted and ground-truth instances are greedily matched in descending face-set IoU;
a match requires class agreement and IoU >= 0.5. Micro precision/recall/F1 and mean matched
IoU are aggregated over parts. Because instances are face *sets*, overlapping assignments are
supported natively on both sides. See `evaluation/instance_common.py` (protocol) and
`evaluation/instance_baseline.py` (connected-component baseline). `evaluation/instance_llm.py`
runs the LLM grouping against any OpenAI-compatible endpoint.

Baseline reference numbers on cross24 (ground-truth labels as input, isolating grouping):

| Method | F1 |
|---|---|
| connected components (rule) | 0.571 |
| classical AAG concave-edge decomposition | 0.714 |
| AAGNet (official weights, matrix clustering) | 0.462 |
| frontier LLMs (gpt-5.5 / gemini-3.1-pro, prompt v2) | **1.000** |

## Notes on paths

The generator and evaluation scripts were released as used in the study and contain absolute
Windows paths near the top of each file; adjust the path constants to your environment before
running. The benchmark data itself is fully self-contained.

## License

- Code (generator, evaluation): MIT License (see `LICENSE`).
- Benchmark data (STEP models, annotations) and LLM responses: CC BY 4.0.

## Citation

Citation information will be added upon acceptance of the accompanying paper.

## Update (2026-09): held-out variants, threshold analysis, and additional runs

The revision adds material that is referenced in the revised manuscript.

### heldout18 — variants generated after the prompt was frozen

`heldout18/` contains 18 parts in three intersection types that do not occur in cross24 and on
which the prompt was never developed (seed 11, zero generator rejects; `generator/gen_heldout_variants.py`):

| Variant | Parts | Faces | Ground-truth structure |
|---|---|---|---|
| `triple_cross_*` | 6 | 26 | three equal-depth through slots; one H-shaped floor shared by **three** instances |
| `oblique_cross_*` | 6 | 18 | two equal-depth through slots crossing at 30-60 deg; non-rectangular shared floor |
| `hole_slot_*` | 6 | 14 | through slot interrupted by a wider vertical through hole (class 1); slot floor and walls severed, no shared face |

Layout mirrors `benchmark/` (`steps/`, `gt_instances/`, `face_labels/`, `face_attributes/`).
LLM responses on this set are in `llm_responses/<model>/heldout18/` (gpt-5.5 and gemini-3.1-pro: F1 1.000 at every IoU threshold).

### Threshold sensitivity and the exclusive-partition oracle

`evaluation/reviewer_analysis.py` reports F1 at IoU thresholds 0.5 / 0.75 / 0.9 / 1.0, a class-agnostic
variant, and an **oracle exclusive partition**: the best F1 any method that assigns each face to exactly
one instance can reach, computed from the ground truth. On cross24 the oracle is 1.000 at tau <= 0.75 and
0.625 at tau >= 0.9 (0.571 on heldout18), so the exclusive representation is binding only under strict
matching; at tau = 0.5 the low scores of partition-based methods are grouping failures (merging or
fragmenting), not representational ones. Results: `supplementary/reviewer_analysis.json`.

### Supplementary runs

- `supplementary/label_source_decomposition/` — gpt-5.5 on mfinst30 with ground-truth face labels and with
  the official AAGNet's predicted labels as Stage 2 input (F1 0.992 / 0.992).
- `supplementary/mfinstseg_hard_subset/` — the 129 MFInstSeg test parts that contain a touching or
  interrupted instance (`mfinstseg_hard_subset.json`, stratification script `evaluation/mfinstseg_hard_subset.py`),
  with gpt-5.5 (F1 0.908) and official-AAGNet (0.965) predictions; connected components: 0.795.
- `supplementary/repeated_runs/` — five additional cross24 runs each for gpt-5.5, gpt-4o and gpt-4o-mini
  (`review_experiments.json` for the per-run F1).

Note: the official AAGNet cannot take external face labels (its semantic and instance heads are predicted
jointly), so on cross24/heldout18 it was run end-to-end with its own predicted labels; the other compared
methods received the ground-truth labels.

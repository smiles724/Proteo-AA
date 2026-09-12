# Stage III Binder Coevolution: Joint Training Results

## Validation overview

The crop-448 run shows a consistent improvement in monomer validation backbone metrics through step 6,000, without the step-6,000 regression observed in the earlier monomer-only Stage III v1 run. Validation loss decreases from 111.0 to 99.93, while Cα RMSD improves from 3.671 Å to 3.446 Å and TM-score increases from 0.7705 to 0.7866.

The crop-512 run is less stable: most geometry metrics regress between steps 2,000 and 4,000. It reached approximately step 5,800 before the wall-time limit, so no step-6,000 validation result is available.

Neither run encountered another CUDA OOM after enabling expandable CUDA allocator segments. Both stopped normally at the 23:50 Slurm time limit.

### Crop 448

| **Step** | **Val loss** | **AA CE / acc.** | **SC local** | **BB post** |
| -------- | ------------ | ---------------- | ------------ | ----------- |
| 2,000    | 111.0        | 2.805 / 0.1285   | 1.816        | 21.15       |
| 4,000    | 107.3        | 2.801 / 0.1302   | 1.836        | 19.90       |
| 6,000    | 99.93        | 2.799 / 0.1310   | 1.818        | 17.69       |

### Crop 512

| **Step** | **Val loss** | **AA CE / acc.** | **SC local** | **BB post** |
| -------- | ------------ | ---------------- | ------------ | ----------- |
| 2,000    | 113.6        | 2.805 / 0.1279   | 1.820        | 21.68       |
| 4,000    | 115.4        | 2.803 / 0.1273   | 1.806        | 21.55       |

## Backbone

The crop-448 run improves consistently from steps 2,000 to 6,000. MSE decreases by 8.9%, Cα and backbone RMSD decrease by approximately 6%, and BB-post loss decreases by 16%. Distogram CE also improves from 1.818 to 1.473, although it remains above the 1.342 reached by the monomer-only Stage III v1 run at step 6,000.

The lDDT loss does not show a corresponding improvement and remains approximately flat.

### Crop 448

| **Step** | **MSE** | **Cα / BB RMSD** | **TM** | **lDDT loss** | **Distogram CE** |
| -------- | ------- | ---------------- | ------ | ------------- | ---------------- |
| 2,000    | 21.25   | 3.671 / 3.621    | 0.7705 | 0.1833        | 1.818            |
| 4,000    | 20.64   | 3.566 / 3.518    | 0.7751 | 0.1885        | 1.684            |
| 6,000    | 19.35   | 3.446 / 3.398    | 0.7866 | 0.1858        | 1.473            |

### Crop 512

| **Step** | **MSE** | **Cα / BB RMSD** | **TM** | **lDDT loss** | **Distogram CE** |
| -------- | ------- | ---------------- | ------ | ------------- | ---------------- |
| 2,000    | 21.76   | 3.625 / 3.574    | 0.7741 | 0.1874        | 1.869            |
| 4,000    | 22.24   | 3.736 / 3.687    | 0.7650 | 0.1832        | 1.609            |

## Comparison with monomer-only Stage III v1

At step 6,000, the crop-448 mixed-data run has lower validation loss, lower backbone RMSD, and higher TM-score than the previous monomer-only Stage III v1 run. Unlike Stage III v1, it does not regress after step 4,000.

| **Metric at step 6,000** | **Monomer Stage III v1** | **Binder crop 448** |
| ------------------------ | ------------------------ | ------------------- |
| Val loss                 | 105.7                    | 99.93               |
| AA CE / accuracy         | 2.796 / 0.1310           | 2.799 / 0.1310      |
| SC local                 | 3.295                    | 1.818               |
| BB post                  | 18.11                    | 17.69               |
| MSE                      | 20.33                    | 19.35               |
| Cα / BB RMSD             | 3.581 / 3.533            | 3.446 / 3.398       |
| TM-score                 | 0.7628                   | 0.7866              |
| lDDT loss                | 0.1806                   | 0.1858              |
| Distogram CE             | 1.342                    | 1.473               |

This comparison is not strictly controlled: Stage III v1 used 308 validation proteins with crop 384, whereas the binder runs used 128 validation proteins with crops 448 or 512.

## Interpretation

AA head: no meaningful improvement is visible. For crop 448, accuracy changes only from 0.1285 to 0.1310 and CE decreases from 2.805 to 2.799. This is similar to the flat AA-head behavior observed in monomer-only Stage III v1.

Side chain: SC-local loss remains flat at approximately 1.82 throughout the crop-448 run. The lower absolute value relative to the approximately 3.3 reported for Stage III v1 is not directly interpretable because the validation set and crop size differ.

Backbone: crop 448 shows the clearest positive result. Most backbone metrics improve continuously through step 6,000, despite the curriculum progressively increasing the complex-data fraction. At step 6,000, approximately 42% of training samples are monomers and 58% are complexes, suggesting that complex training has not yet caused measurable degradation of the monomer fold prior.

Crop size: crop 448 is currently preferable operationally and empirically. It completes approximately 7% more steps per day than crop 512, avoids OOM, and shows a more stable monomer-validation trajectory. However, crop 512 preserves more complete interfaces, so the final choice should be based on binder-specific validation rather than these monomer metrics.

## Limitation and next evaluation

These `val_*` metrics evaluate monomers, not binder chains. They can show whether the fold prior is preserved, but they cannot establish that binder generation has improved. Checkpoint selection should therefore use the non-redundant PINDER binder validation split.

The crop-448 checkpoints at steps 4,000, 5,000, and 6,000 should be evaluated with the same PINDER binder-backbone protocol before selecting a continuation or final checkpoint.
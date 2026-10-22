# Phase 2: probes, DAS, and edge attribution patching on InterpBench models

## Linear probes (readable = held-out accuracy >= 0.95)

| Case | Model | Variable | First write | First readable | Readable before write | Position-determined | Max acc | Stages |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | compiled | length_5 | 2 | 0 | yes | yes | 1.000 | `111111111` |
| 2 | compiled | opp_idx_4 | 4 | 0 | yes | yes | 1.000 | `111111111` |
| 2 | compiled | opp_idx-1_3 | 6 | 0 | yes | yes | 1.000 | `111111111` |
| 2 | compiled | reverse_1 | 7 | 7 | no | no | 1.000 | `000000011` |
| 2 | trained | length_5 | 2 | 0 | yes | yes | 1.000 | `111111111` |
| 2 | trained | opp_idx_4 | 4 | 0 | yes | yes | 1.000 | `111111111` |
| 2 | trained | opp_idx-1_3 | 6 | 0 | yes | yes | 1.000 | `111111111` |
| 2 | trained | reverse_1 | 7 | 5 | yes | no | 1.000 | `000001111` |
| 7 | compiled | hist_7 | 2 | 2 | no | no | 1.000 | `001` |
| 7 | trained | hist_7 | 2 | 2 | no | no | 1.000 | `00111` |
| 13 | compiled | shift_by(-1)_13 | 1 | 1 | no | no | 1.000 | `01111` |
| 13 | compiled | shift_by(1)_12 | 1 | 1 | no | no | 1.000 | `01111` |
| 13 | compiled | sequence_map_10 | 2 | 1 | yes | no | 1.000 | `01111` |
| 13 | compiled | sequence_map_11 | 2 | 1 | yes | no | 1.000 | `01111` |
| 13 | compiled | sequence_map_9 | 4 | 1 | yes | no | 1.000 | `01111` |
| 13 | trained | shift_by(-1)_13 | 1 | 1 | no | no | 1.000 | `01111` |
| 13 | trained | shift_by(1)_12 | 1 | 1 | no | no | 1.000 | `01110` |
| 13 | trained | sequence_map_10 | 2 | 1 | yes | no | 1.000 | `01111` |
| 13 | trained | sequence_map_11 | 2 | 1 | yes | no | 0.997 | `01111` |
| 13 | trained | sequence_map_9 | 4 | 1 | yes | no | 1.000 | `01111` |
| 21 | compiled | sequence_map_21 | 2 | 0 | yes | no | 1.000 | `111111111` |
| 21 | compiled | selector_width_19 | 4 | 4 | no | no | 1.000 | `000011111` |
| 21 | compiled | sequence_map_18 | 6 | 3 | yes | no | 1.000 | `000111111` |
| 21 | compiled | unique_tokens_16 | 7 | 0 | yes | no | 1.000 | `111111111` |
| 21 | trained | sequence_map_21 | 2 | 0 | yes | no | 1.000 | `111000000` |
| 21 | trained | selector_width_19 | 4 | 4 | no | no | 0.958 | `000011100` |
| 21 | trained | sequence_map_18 | 6 | 2 | yes | no | 0.961 | `001100000` |
| 21 | trained | unique_tokens_16 | 7 | 0 | yes | no | 1.000 | `111111111` |
| 44 | compiled | count_greater_than_22 | 2 | 2 | no | no | 1.000 | `001` |
| 44 | trained | count_greater_than_22 | 2 | 1 | yes | no | 0.987 | `01111` |
| 124 | compiled | selector_width_27 | 2 | 0 | yes | no | 1.000 | `1111111` |
| 124 | compiled | selector_width_25 | 4 | 0 | yes | no | 1.000 | `1111111` |
| 124 | compiled | map_24 | 6 | 0 | yes | no | 1.000 | `1111111` |
| 124 | trained | selector_width_27 | 2 | 0 | yes | no | 1.000 | `1111111` |
| 124 | trained | selector_width_25 | 4 | 0 | yes | no | 1.000 | `1111111` |
| 124 | trained | map_24 | 6 | 0 | yes | no | 1.000 | `1111111` |

Share of input-dependent variables linearly readable before their first write: compiled 60%, trained 73%. Position-determined variables (constant at fixed length) are excluded; they read from the position embedding at stage 0 and cannot be intervened on.

## DAS (aligned = IIA >= 0.9)

| Case | Model | Variable | First write | First aligned | IIA at write | Max IIA | Random max | Base rate | IIA at write, changing pairs | Max IIA, changing | Random, changing | Changing pairs | Stages |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | compiled | reverse_1 | 7 | 7 | 1.000 | 1.000 | 0.020 | 0.000 | 1.000 | 1.000 | 0.020 | 256 | `000000011` |
| 2 | trained | reverse_1 | 7 | 7 | 1.000 | 1.000 | 0.223 | 0.000 | 1.000 | 1.000 | 0.223 | 256 | `000000011` |
| 7 | compiled | hist_7 | 2 | 1 | 1.000 | 1.000 | 0.148 | 0.000 | 1.000 | 1.000 | 0.148 | 256 | `011` |
| 7 | trained | hist_7 | 2 | 1 | 0.949 | 1.000 | 0.434 | 0.000 | 0.949 | 1.000 | 0.434 | 256 | `01111` |
| 13 | compiled | shift_by(-1)_13 | 1 | 1 | 1.000 | 1.000 | 0.625 | 0.625 | 1.000 | 1.000 | 0.062 | 96 | `01000` |
| 13 | compiled | shift_by(1)_12 | 1 | 1 | 1.000 | 1.000 | 0.758 | 0.750 | 1.000 | 1.000 | 0.109 | 64 | `01000` |
| 13 | compiled | sequence_map_10 | 2 | 2 | 0.910 | 0.910 | 0.738 | 0.699 | 0.701 | 0.701 | 0.273 | 77 | `00110` |
| 13 | compiled | sequence_map_11 | 2 | 2 | 1.000 | 1.000 | 0.426 | 0.395 | 1.000 | 1.000 | 0.110 | 155 | `00110` |
| 13 | compiled | sequence_map_9 | 4 | 2 | 1.000 | 1.000 | 0.152 | 0.000 | 1.000 | 1.000 | 0.152 | 256 | `00111` |
| 13 | trained | shift_by(-1)_13 | 1 | 1 | 0.992 | 0.992 | 0.625 | 0.625 | 0.979 | 0.979 | 0.062 | 96 | `01000` |
| 13 | trained | shift_by(1)_12 | 1 | 1 | 0.996 | 0.996 | 0.750 | 0.750 | 0.984 | 0.984 | 0.047 | 64 | `01000` |
| 13 | trained | sequence_map_10 | 2 | none | 0.812 | 0.816 | 0.699 | 0.699 | 0.403 | 0.468 | 0.065 | 77 | `00000` |
| 13 | trained | sequence_map_11 | 2 | none | 0.840 | 0.840 | 0.422 | 0.395 | 0.774 | 0.787 | 0.077 | 155 | `00000` |
| 13 | trained | sequence_map_9 | 4 | 4 | 1.000 | 1.000 | 0.074 | 0.000 | 1.000 | 1.000 | 0.074 | 256 | `00001` |
| 21 | compiled | sequence_map_21 | 2 | 0 | 0.977 | 1.000 | 0.863 | 0.852 | 0.921 | 1.000 | 0.737 | 38 | `111011100` |
| 21 | compiled | selector_width_19 | 4 | 0 | 1.000 | 1.000 | 0.949 | 0.977 | 1.000 | 1.000 | 0.500 | 6 | `111111111` |
| 21 | compiled | sequence_map_18 | 6 | 0 | 1.000 | 1.000 | 0.848 | 0.902 | 1.000 | 1.000 | 0.720 | 25 | `111011111` |
| 21 | compiled | unique_tokens_16 | 7 | 0 | 1.000 | 1.000 | 0.328 | 0.000 | 1.000 | 1.000 | 0.328 | 256 | `111111111` |
| 21 | trained | sequence_map_21 | 2 | 0 | 0.848 | 0.957 | 0.891 | 0.852 | 0.000 | 0.737 | 0.526 | 38 | `100000000` |
| 21 | trained | selector_width_19 | 4 | 0 | 0.980 | 0.992 | 0.977 | 0.977 | 0.333 | 0.667 | 0.167 | 6 | `111111111` |
| 21 | trained | sequence_map_18 | 6 | 1 | 0.922 | 0.941 | 0.914 | 0.902 | 0.200 | 0.400 | 0.400 | 25 | `011111111` |
| 21 | trained | unique_tokens_16 | 7 | 0 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 256 | `100011111` |
| 44 | compiled | count_greater_than_22 | 2 | 1 | 1.000 | 1.000 | 0.109 | 0.000 | 1.000 | 1.000 | 0.109 | 256 | `011` |
| 44 | trained | count_greater_than_22 | 2 | 1 | 0.348 | 1.000 | 0.355 | 0.000 | 0.348 | 1.000 | 0.355 | 256 | `01011` |
| 124 | compiled | selector_width_27 | 2 | 2 | 1.000 | 1.000 | 0.508 | 0.508 | 1.000 | 1.000 | 0.048 | 126 | `0010000` |
| 124 | compiled | selector_width_25 | 4 | 2 | 1.000 | 1.000 | 0.191 | 0.000 | 1.000 | 1.000 | 0.191 | 256 | `0011111` |
| 124 | compiled | map_24 | 6 | 1 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 256 | `0111111` |
| 124 | trained | selector_width_27 | 2 | none | 0.445 | 0.508 | 0.508 | 0.508 | 0.000 | 0.000 | 0.000 | 126 | `0000000` |
| 124 | trained | selector_width_25 | 4 | 4 | 1.000 | 1.000 | 0.633 | 0.000 | 1.000 | 1.000 | 0.633 | 256 | `0000111` |
| 124 | trained | map_24 | 6 | 4 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | 256 | `0000111` |

Means over variables: compiled: max IIA 0.994, IIA at first write 0.992, random 0.444; on output-changing pairs: at first write 0.975, max 0.980, random 0.234; trained: max IIA 0.936, IIA at first write 0.876, random 0.500; on output-changing pairs: at first write 0.665, max 0.801, random 0.204

Base rate is the share of test pairs whose program counterfactual equals the unpatched output; IIA near the base rate is uninformative. The changing-pair columns exclude those pairs.

## Geometry detectors against probe and DAS stage events

| Model | Truth | Detector | mean F1 | mean all-positive F1 |
| --- | --- | --- | --- | --- |
| compiled | das_aligned | change | 0.506 | 0.454 |
| compiled | das_aligned | cka | 0.244 | 0.454 |
| compiled | das_aligned | pr | 0.534 | 0.454 |
| compiled | probe_readable | change | 0.456 | 0.393 |
| compiled | probe_readable | cka | 0.428 | 0.393 |
| compiled | probe_readable | pr | 0.484 | 0.393 |
| trained | das_aligned | change | 0.366 | 0.366 |
| trained | das_aligned | cka | 0.449 | 0.366 |
| trained | das_aligned | pr | 0.471 | 0.366 |
| trained | probe_readable | change | 0.304 | 0.304 |
| trained | probe_readable | cka | 0.381 | 0.304 |
| trained | probe_readable | pr | 0.411 | 0.304 |

## Edge attribution patching (trained models)

| Case | Edges: AUROC | AP | P@k | k | Nodes: AUROC | P@k | k | Truth scorable / internal / out |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 0.705 | 0.224 | 0.200 | 10 | 0.819 | 0.714 | 7 | 10 / 15 / 2 |
| 7 | 0.998 | 0.950 | 0.750 | 4 | 1.000 | 1.000 | 3 | 4 / 7 / 0 |
| 13 | 0.605 | 0.442 | 0.375 | 16 | 0.833 | 0.833 | 6 | 16 / 6 / 0 |
| 21 | 0.458 | 0.228 | 0.286 | 14 | 0.800 | 0.800 | 5 | 14 / 3 / 0 |
| 44 | 1.000 | 1.000 | 1.000 | 4 | 1.000 | 1.000 | 3 | 4 / 7 / 0 |
| 124 | 0.845 | 0.490 | 0.444 | 9 | 0.686 | 0.571 | 7 | 9 / 15 / 0 |

Means: edge AUROC 0.769, edge P@k 0.509, node AUROC 0.856, node P@k 0.820

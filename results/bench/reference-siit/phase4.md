# Phase 4: SIIT models for the twelve RaspFormer programs

## SIIT training outcome

| Program | Held-out token acc | Mean IIA | Mean SIIA |
| --- | --- | --- | --- |
| P01_absolute | 1.000 | 1.000 | 1.000 |
| P02_index_parity | 1.000 | 1.000 | 1.000 |
| P03_increment_by_index | 1.000 | 1.000 | 1.000 |
| P04_first_element | 1.000 | 1.000 | 1.000 |
| P05_histogram | 0.991 | 0.980 | 0.990 |
| P06_count_greater_than | 1.000 | 1.000 | 1.000 |
| P07_sum_with_next | 1.000 | 1.000 | 1.000 |
| P08_pairwise_sum | 1.000 | 1.000 | 1.000 |
| P09_check_increasing | 0.979 | 0.983 | 0.994 |
| P10_rotate_left | 1.000 | 1.000 | 1.000 |
| P11_check_palindrome | 0.990 | 0.993 | 0.992 |
| P12_sorting | 1.000 | 0.971 | 0.975 |

## Blind first-write detection (mean F1 over programs)

| Model | Truth | Detector | mean F1 | mean all-positive F1 |
| --- | --- | --- | --- | --- |
| compiled | schedule | change | 0.882 | 0.706 |
| compiled | schedule | cka | 0.897 | 0.706 |
| compiled | schedule | pr | 0.876 | 0.706 |
| trained | circuit | change | 0.821 | 0.821 |
| trained | circuit | cka | 0.800 | 0.821 |
| trained | circuit | pr | 0.836 | 0.821 |
| trained | schedule | change | 0.706 | 0.706 |
| trained | schedule | cka | 0.700 | 0.706 |
| trained | schedule | pr | 0.720 | 0.706 |

## Stress suite (trained models, schedule truth, means over programs)

| Transform | Fraction | Detector | mean token acc | mean F1 |
| --- | --- | --- | --- | --- |
| compression-pca | 0.25 | change | 0.485 | 0.706 |
| compression-pca | 0.25 | cka | 0.485 | 0.726 |
| compression-pca | 0.25 | pr | 0.485 | 0.727 |
| compression-pca | 0.5 | change | 0.693 | 0.706 |
| compression-pca | 0.5 | cka | 0.693 | 0.703 |
| compression-pca | 0.5 | pr | 0.693 | 0.681 |
| compression-pca | 0.75 | change | 0.838 | 0.706 |
| compression-pca | 0.75 | cka | 0.838 | 0.700 |
| compression-pca | 0.75 | pr | 0.838 | 0.706 |
| compression-pca | 1.0 | change | 0.997 | 0.706 |
| compression-pca | 1.0 | cka | 0.997 | 0.700 |
| compression-pca | 1.0 | pr | 0.997 | 0.720 |
| compression-random | 0.5 | change | 0.414 | 0.706 |
| compression-random | 0.5 | cka | 0.414 | 0.708 |
| compression-random | 0.5 | pr | 0.414 | 0.718 |
| rotation | 1.0 | change | 1.000 | 0.706 |
| rotation | 1.0 | cka | 1.000 | 0.700 |
| rotation | 1.0 | pr | 1.000 | 0.720 |

## Probes and DAS

Share of input-dependent variables readable before first write: compiled 26%, trained 21%

DAS on output-changing pairs, means over variables: compiled: at first write 0.683, max 0.767, random 0.141; trained: at first write 0.625, max 0.643, random 0.094

## Edge attribution patching (trained models)

| Program | Edge AUROC | Edge P@k | Node AUROC | Node P@k | k edges |
| --- | --- | --- | --- | --- | --- |
| P01_absolute | 1.000 | 1.000 | 1.000 | 1.000 | 2 |
| P02_index_parity | 0.564 | 0.333 | 0.333 | 0.333 | 3 |
| P03_increment_by_index | 0.730 | 0.667 | 0.667 | 0.667 | 3 |
| P04_first_element | 0.591 | 0.500 | 0.667 | 0.667 | 4 |
| P05_histogram | 0.985 | 0.750 | 1.000 | 1.000 | 4 |
| P06_count_greater_than | 1.000 | 1.000 | 1.000 | 1.000 | 4 |
| P07_sum_with_next | 0.699 | 0.455 | 0.671 | 0.571 | 11 |
| P08_pairwise_sum | 0.724 | 0.455 | 0.700 | 0.714 | 11 |
| P09_check_increasing | 0.660 | 0.400 | 0.843 | 0.857 | 10 |
| P10_rotate_left | 0.643 | 0.333 | 0.636 | 0.667 | 9 |
| P11_check_palindrome | 0.740 | 0.125 | 0.827 | 0.727 | 16 |
| P12_sorting | 0.754 | 0.333 | 0.818 | 0.833 | 9 |

## Blind program recovery

| Program | Model | Nodes | Ops | Output agreement | Coverage | Counterfactual agreement | Ablation agreement | Variables exact / missing |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P01_absolute | compiled | 2 | `0.result[0]:select,0.mlp_out:mlp` | 1.000 | 1.000 | 1.000 | 1.000 (0 outside) | 1 / 0 |
| P01_absolute | trained | 1 | `0.mlp_out:mlp` | 1.000 | 1.000 | 1.000 | 1.000 (4 outside) | 1 / 0 |
| P02_index_parity | compiled | 2 | `0.result[0]:select,0.mlp_out:mlp` | 1.000 | 1.000 | n/a | 1.000 (0 outside) | 1 / 0 |
| P02_index_parity | trained | 1 | `0.mlp_out:mlp` | 1.000 | 1.000 | n/a | 1.000 (4 outside) | 1 / 0 |
| P03_increment_by_index | compiled | 2 | `0.result[0]:select,0.mlp_out:mlp` | 1.000 | 1.000 | 1.000 | 1.000 (0 outside) | 1 / 0 |
| P03_increment_by_index | trained | 1 | `0.mlp_out:mlp` | 1.000 | 1.000 | 1.000 | 0.562 (4 outside) | 1 / 0 |
| P04_first_element | compiled | 2 | `0.result[0]:select,0.mlp_out:mlp` | 1.000 | 1.000 | 1.000 | 1.000 (0 outside) | 1 / 0 |
| P04_first_element | trained | 1 | `0.result[0]:select` | 1.000 | 1.000 | 1.000 | 1.000 (4 outside) | 1 / 0 |
| P05_histogram | compiled | 2 | `0.result[0]:select,0.mlp_out:mlp` | 0.970 | 0.970 | 0.986 | 1.000 (0 outside) | 1 / 0 |
| P05_histogram | trained | 2 | `0.result[0]:select,0.mlp_out:mlp` | 0.941 | 0.941 | 0.984 | 0.991 (3 outside) | 1 / 0 |
| P06_count_greater_than | compiled | 2 | `0.result[0]:select,0.mlp_out:mlp` | 0.974 | 0.974 | 1.000 | 1.000 (0 outside) | 1 / 0 |
| P06_count_greater_than | trained | 2 | `0.result[0]:select,0.mlp_out:mlp` | 0.628 | 0.752 | 0.329 | 1.000 (3 outside) | 1 / 0 |
| P07_sum_with_next | compiled | 6 | `0.result[0]:select,0.mlp_out:mlp,1.result[0]:select,1.mlp_out:mlp,2.result[0]:select,2.mlp_out:mlp` | 0.809 | 0.809 | 1.000 | 1.000 (0 outside) | 4 / 0 |
| P07_sum_with_next | trained | 5 | `0.result[0]:select,0.mlp_out:mlp,1.mlp_out:mlp,2.result[0]:select,2.mlp_out:mlp` | 0.267 | 0.318 | 0.775 | 0.688 (10 outside) | 2 / 0 |
| P08_pairwise_sum | compiled | 6 | `0.result[0]:select,0.mlp_out:mlp,1.result[0]:select,1.mlp_out:mlp,2.result[0]:select,2.mlp_out:mlp` | 0.786 | 0.786 | 1.000 | 1.000 (0 outside) | 4 / 0 |
| P08_pairwise_sum | trained | 5 | `0.result[0]:select,0.mlp_out:mlp,1.mlp_out:mlp,2.result[0]:select,2.mlp_out:mlp` | 0.253 | 0.332 | 0.333 | 0.936 (10 outside) | 4 / 0 |
| P09_check_increasing | compiled | 6 | `0.result[0]:select,0.mlp_out:mlp,1.result[0]:select,1.mlp_out:mlp,2.result[0]:select,2.mlp_out:mlp` | 0.979 | 1.000 | n/a | 1.000 (0 outside) | 1 / 3 |
| P09_check_increasing | trained | 5 | `0.result[0]:select,0.mlp_out:mlp,1.result[0]:select,1.mlp_out:mlp,2.mlp_out:mlp` | 1.000 | 1.000 | n/a | 1.000 (10 outside) | 1 / 3 |
| P10_rotate_left | compiled | 6 | `0.result[0]:select,0.mlp_out:mlp,1.result[0]:select,1.mlp_out:mlp,2.result[0]:select,2.mlp_out:mlp` | 1.000 | 1.000 | 1.000 | 1.000 (0 outside) | 3 / 0 |
| P10_rotate_left | trained | 4 | `0.result[0]:select,0.mlp_out:mlp,1.mlp_out:mlp,2.result[0]:select` | 0.974 | 1.000 | 0.742 | 1.000 (11 outside) | 3 / 0 |
| P11_check_palindrome | compiled | 12 | `0.result[0]:select,0.mlp_out:mlp,1.result[0]:select,1.mlp_out:mlp,2.result[0]:select,2.mlp_out:mlp,3.result[0]:select,3.mlp_out:mlp,4.result[0]:select,4.mlp_out:mlp,5.result[0]:select,5.mlp_out:mlp` | 0.990 | 1.000 | n/a | 1.000 (0 outside) | 2 / 5 |
| P11_check_palindrome | trained | 9 | `0.result[0]:select,0.mlp_out:mlp,1.mlp_out:mlp,2.mlp_out:mlp,3.result[0]:select,3.mlp_out:mlp,4.result[0]:select,4.mlp_out:mlp,5.mlp_out:mlp` | 1.000 | 1.000 | n/a | 1.000 (21 outside) | 2 / 5 |
| P12_sorting | compiled | 6 | `0.result[0]:select,0.mlp_out:mlp,1.result[0]:select,1.mlp_out:mlp,2.result[0]:select,2.mlp_out:mlp` | 0.056 | 0.059 | n/a | 1.000 (0 outside) | 1 / 2 |
| P12_sorting | trained | 4 | `0.mlp_out:mlp,1.result[0]:select,1.mlp_out:mlp,2.result[0]:select` | 0.000 | 0.000 | n/a | 0.503 (11 outside) | 1 / 2 |

Means: compiled: output agreement 0.880, counterfactual 0.998, variables exact 21/31; trained: output agreement 0.755, counterfactual 0.770, variables exact 19/31

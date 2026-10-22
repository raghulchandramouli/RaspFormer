# Phase 3: blind program recovery and length generalization

## Recovery: three tests per model

| Case | Model | Nodes | Ops | Classes | Output agreement | Coverage | Counterfactual agreement | Ablation agreement | Variables exact / refined / missing |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | compiled | 8 | `0.result[0]:select,0.mlp_out:mlp,1.result[0]:select,1.mlp_out:mlp,2.result[0]:select,2.mlp_out:mlp,3.result[0]:select,3.mlp_out:mlp` | 1,1,1,9,1,9,26,1 | 1.000 | 1.000 | 1.000 | 1.000 (0 outside) | 4 / 0 / 0 |
| 2 | trained | 5 | `0.result[0]:select,0.mlp_out:mlp,1.mlp_out:mlp,2.mlp_out:mlp,3.result[0]:select` | 1,11,14,25,26 | 0.038 | 0.537 | 0.000 | 0.978 (15 outside) | 2 / 2 / 0 |
| 7 | compiled | 2 | `0.result[0]:select,0.mlp_out:mlp` | 7,7 | 1.000 | 1.000 | 1.000 | 1.000 (0 outside) | 1 / 0 / 0 |
| 7 | trained | 2 | `0.result[0]:select,0.mlp_out:mlp` | 9,7 | 0.737 | 1.000 | 0.193 | 0.953 (8 outside) | 1 / 0 / 0 |
| 13 | compiled | 6 | `0.result[0]:select,0.result[1]:select,0.mlp_out:mlp,1.result[0]:select,1.result[1]:select,1.mlp_out:mlp` | 3,3,3,1,1,3 | 0.971 | 1.000 | 0.772 | 1.000 (0 outside) | 3 / 0 / 2 |
| 13 | trained | 4 | `0.result[0]:select,0.result[1]:select,0.mlp_out:mlp,1.mlp_out:mlp` | 3,3,3,3 | 0.988 | 0.993 | 0.959 | 1.000 (6 outside) | 3 / 0 / 2 |
| 21 | compiled | 8 | `0.result[0]:select,0.mlp_out:mlp,1.result[0]:bag,1.mlp_out:mlp,2.result[0]:select,2.mlp_out:mlp,3.result[0]:bag,3.mlp_out:mlp` | 1,3,2,2,1,1,3,1 | 0.506 | 0.969 | 0.003 | 1.000 (0 outside) | 1 / 0 / 3 |
| 21 | trained | 3 | `1.mlp_out:mlp,2.mlp_out:mlp,3.result[0]:bag` | 2,2,3 | 0.000 | 0.000 | 0.000 | 0.266 (17 outside) | 1 / 0 / 3 |
| 44 | compiled | 2 | `0.result[0]:select,0.mlp_out:mlp` | 9,9 | 1.000 | 1.000 | 1.000 | 1.000 (0 outside) | 1 / 0 / 0 |
| 44 | trained | 2 | `0.result[0]:select,0.mlp_out:mlp` | 9,9 | 0.865 | 1.000 | 0.350 | 0.892 (8 outside) | 1 / 0 / 0 |
| 124 | compiled | 6 | `0.result[0]:select,0.mlp_out:mlp,1.result[0]:select,1.mlp_out:mlp,2.result[0]:select,2.mlp_out:mlp` | 2,2,2,2,1,2 | 1.000 | 1.000 | 1.000 | 1.000 (0 outside) | 3 / 0 / 0 |
| 124 | trained | 5 | `0.result[0]:select,0.mlp_out:mlp,1.result[0]:select,1.mlp_out:mlp,2.mlp_out:mlp` | 1,1,2,2,2 | 0.000 | 0.000 | n/a | 1.000 (10 outside) | 3 / 0 / 0 |

Means: compiled: output agreement 0.913, counterfactual agreement 0.796, variables exact 13/18, bag heads 2; trained: output agreement 0.438, counterfactual agreement 0.300, variables exact 11/18, bag heads 1

Output agreement counts undefined positions as disagreement. A bag head is one whose attention pattern was not crisp enough for a selector, so its output is tabulated against the whole key multiset instead.

## Length generalization: InterpBench trained models (trained at nine tokens)

| Case | Tokens | Model token acc | Model seq acc | Program coverage | Program = model | Model acc where program defined | Model acc where program undefined |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4 | 0.031 | 0.000 | 0.025 | 0.100 | 0.050 | 0.031 |
| 2 | 5 | 0.196 | 0.000 | 0.188 | 0.021 | 0.037 | 0.233 |
| 2 | 6 | 0.173 | 0.000 | 0.289 | 0.049 | 0.069 | 0.216 |
| 2 | 7 | 0.109 | 0.000 | 0.471 | 0.045 | 0.033 | 0.176 |
| 2 | 8 | 0.062 | 0.000 | 0.507 | 0.048 | 0.028 | 0.098 |
| 2 | 9 | 1.000 | 1.000 | 0.546 | 0.057 | 1.000 | 1.000 |
| 7 | 4 | 0.575 | 0.135 | 1.000 | 0.720 | 0.575 | n/a |
| 7 | 5 | 0.628 | 0.040 | 1.000 | 0.655 | 0.628 | n/a |
| 7 | 6 | 0.643 | 0.040 | 1.000 | 0.690 | 0.643 | n/a |
| 7 | 7 | 0.628 | 0.100 | 0.649 | 0.945 | 0.945 | 0.043 |
| 7 | 8 | 0.845 | 0.550 | 0.667 | 0.963 | 0.969 | 0.597 |
| 7 | 9 | 1.000 | 1.000 | 1.000 | 0.761 | 1.000 | n/a |
| 13 | 4 | 0.940 | 0.825 | 0.500 | 1.000 | 1.000 | 0.880 |
| 13 | 5 | 0.993 | 0.965 | 0.600 | 1.000 | 1.000 | 0.983 |
| 13 | 6 | 0.981 | 0.890 | 0.667 | 1.000 | 1.000 | 0.943 |
| 13 | 7 | 0.977 | 0.840 | 0.714 | 1.000 | 1.000 | 0.920 |
| 13 | 8 | 0.974 | 0.815 | 0.750 | 1.000 | 1.000 | 0.897 |
| 13 | 9 | 1.000 | 1.000 | 0.982 | 0.995 | 1.000 | 1.000 |
| 21 | 4 | 1.000 | 1.000 | 0.000 | n/a | n/a | 1.000 |
| 21 | 5 | 1.000 | 1.000 | 0.000 | n/a | n/a | 1.000 |
| 21 | 6 | 1.000 | 1.000 | 0.000 | n/a | n/a | 1.000 |
| 21 | 7 | 1.000 | 1.000 | 0.000 | n/a | n/a | 1.000 |
| 21 | 8 | 1.000 | 1.000 | 0.000 | n/a | n/a | 1.000 |
| 21 | 9 | 1.000 | 1.000 | 0.000 | n/a | n/a | 1.000 |
| 44 | 4 | 0.901 | 0.665 | 1.000 | 0.824 | 0.901 | n/a |
| 44 | 5 | 0.923 | 0.680 | 1.000 | 0.844 | 0.923 | n/a |
| 44 | 6 | 0.897 | 0.520 | 1.000 | 0.808 | 0.897 | n/a |
| 44 | 7 | 0.893 | 0.430 | 1.000 | 0.795 | 0.893 | n/a |
| 44 | 8 | 0.902 | 0.420 | 1.000 | 0.811 | 0.902 | n/a |
| 44 | 9 | 1.000 | 1.000 | 1.000 | 0.823 | 1.000 | n/a |
| 124 | 4 | 0.990 | 0.990 | 0.000 | n/a | n/a | 0.990 |
| 124 | 5 | 0.995 | 0.995 | 0.000 | n/a | n/a | 0.995 |
| 124 | 6 | 0.993 | 0.985 | 0.000 | n/a | n/a | 0.993 |
| 124 | 7 | 0.990 | 0.990 | 0.000 | n/a | n/a | 0.990 |
| 124 | 8 | 1.000 | 1.000 | 0.000 | n/a | n/a | 1.000 |
| 124 | 9 | 1.000 | 1.000 | 0.000 | n/a | n/a | 1.000 |

## Length generalization: RaspFormer learned models (trained at six tokens)

| Program | Tokens | Mean token acc over seeds | Mean seq acc |
| --- | --- | --- | --- |
| P01_absolute | 3 | 1.000 | 1.000 |
| P01_absolute | 4 | 1.000 | 1.000 |
| P01_absolute | 5 | 1.000 | 1.000 |
| P01_absolute | 6 | 1.000 | 1.000 |
| P02_index_parity | 3 | 1.000 | 1.000 |
| P02_index_parity | 4 | 1.000 | 1.000 |
| P02_index_parity | 5 | 1.000 | 1.000 |
| P02_index_parity | 6 | 1.000 | 1.000 |
| P03_increment_by_index | 3 | 0.964 | 0.898 |
| P03_increment_by_index | 4 | 0.989 | 0.956 |
| P03_increment_by_index | 5 | 1.000 | 1.000 |
| P03_increment_by_index | 6 | 1.000 | 1.000 |
| P04_first_element | 3 | 1.000 | 1.000 |
| P04_first_element | 4 | 1.000 | 1.000 |
| P04_first_element | 5 | 1.000 | 1.000 |
| P04_first_element | 6 | 1.000 | 1.000 |
| P05_histogram | 3 | 0.142 | 0.021 |
| P05_histogram | 4 | 0.519 | 0.204 |
| P05_histogram | 5 | 0.922 | 0.828 |
| P05_histogram | 6 | 1.000 | 1.000 |
| P06_count_greater_than | 3 | 0.387 | 0.014 |
| P06_count_greater_than | 4 | 0.416 | 0.008 |
| P06_count_greater_than | 5 | 0.607 | 0.040 |
| P06_count_greater_than | 6 | 1.000 | 1.000 |
| P07_sum_with_next | 3 | 0.996 | 0.989 |
| P07_sum_with_next | 4 | 0.996 | 0.984 |
| P07_sum_with_next | 5 | 1.000 | 1.000 |
| P07_sum_with_next | 6 | 1.000 | 0.998 |
| P08_pairwise_sum | 3 | 0.702 | 0.107 |
| P08_pairwise_sum | 4 | 0.783 | 0.133 |
| P08_pairwise_sum | 5 | 0.840 | 0.198 |
| P08_pairwise_sum | 6 | 1.000 | 1.000 |
| P09_check_increasing | 3 | 0.924 | 0.914 |
| P09_check_increasing | 4 | 0.960 | 0.957 |
| P09_check_increasing | 5 | 0.986 | 0.983 |
| P09_check_increasing | 6 | 0.992 | 0.991 |
| P10_rotate_left | 3 | 0.710 | 0.131 |
| P10_rotate_left | 4 | 0.786 | 0.146 |
| P10_rotate_left | 5 | 0.840 | 0.200 |
| P10_rotate_left | 6 | 1.000 | 1.000 |
| P11_check_palindrome | 3 | 0.664 | 0.636 |
| P11_check_palindrome | 4 | 0.941 | 0.929 |
| P11_check_palindrome | 5 | 0.951 | 0.940 |
| P11_check_palindrome | 6 | 0.987 | 0.987 |
| P12_sorting | 3 | 0.456 | 0.010 |
| P12_sorting | 4 | 0.521 | 0.013 |
| P12_sorting | 5 | 0.730 | 0.081 |
| P12_sorting | 6 | 1.000 | 1.000 |

Mean token accuracy over all 36 models by length: 3 tokens 0.745, 4 tokens 0.826, 5 tokens 0.906, 6 tokens 0.998


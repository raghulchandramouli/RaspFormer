# Phase 3: blind program recovery and length generalization

## Recovery: three tests per model

| Case | Model | Nodes | Ops | Classes | Output agreement | Coverage | Counterfactual agreement | Ablation agreement | Variables exact / refined / missing |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | compiled | 8 | `0.result[0]:select,0.mlp_out:mlp,1.result[0]:select,1.mlp_out:mlp,2.result[0]:select,2.mlp_out:mlp,3.result[0]:select,3.mlp_out:mlp` | 1,1,1,9,1,9,26,1 | 0.986 | 0.986 | 1.000 | 1.000 (0 outside) | 4 / 0 / 0 |
| 2 | trained | 5 | `0.result[0]:select,0.mlp_out:mlp,1.mlp_out:mlp,2.mlp_out:mlp,3.result[0]:select` | 1,22,14,28,26 | 0.000 | 0.000 | n/a | 0.978 (15 outside) | 2 / 2 / 0 |
| 7 | compiled | 2 | `0.result[0]:select,0.mlp_out:mlp` | 7,7 | 1.000 | 1.000 | 1.000 | 1.000 (0 outside) | 1 / 0 / 0 |
| 7 | trained | 2 | `0.result[0]:select,0.mlp_out:mlp` | 10,7 | 0.726 | 0.988 | 0.194 | 0.953 (8 outside) | 1 / 0 / 0 |
| 13 | compiled | 6 | `0.result[0]:select,0.result[1]:select,0.mlp_out:mlp,1.result[0]:select,1.result[1]:select,1.mlp_out:mlp` | 4,3,3,1,1,3 | 0.956 | 0.985 | 0.722 | 1.000 (0 outside) | 3 / 0 / 2 |
| 13 | trained | 4 | `0.result[1]:select,0.result[0]:select,0.mlp_out:mlp,1.mlp_out:mlp` | 3,3,3,3 | 0.966 | 0.971 | 0.925 | 1.000 (6 outside) | 3 / 0 / 2 |
| 21 | compiled | 8 | `0.result[0]:select,0.mlp_out:mlp,1.result[0]:select,1.mlp_out:mlp,2.result[0]:select,2.mlp_out:mlp,3.result[0]:bag,3.mlp_out:mlp` | 1,2,2,2,1,1,3,1 | 0.506 | 0.969 | 0.003 | 1.000 (0 outside) | 1 / 0 / 3 |
| 21 | trained | 3 | `1.mlp_out:mlp,2.mlp_out:mlp,3.result[0]:select` | 2,2,3 | 0.594 | 0.757 | 0.000 | 0.266 (17 outside) | 1 / 0 / 3 |
| 44 | compiled | 2 | `0.result[0]:select,0.mlp_out:mlp` | 9,9 | 0.990 | 0.990 | 1.000 | 1.000 (0 outside) | 1 / 0 / 0 |
| 44 | trained | 2 | `0.result[0]:select,0.mlp_out:mlp` | 9,9 | 0.854 | 0.910 | 0.810 | 0.892 (8 outside) | 1 / 0 / 0 |
| 124 | compiled | 6 | `0.result[0]:select,0.mlp_out:mlp,1.result[0]:select,1.mlp_out:mlp,2.result[0]:select,2.mlp_out:mlp` | 2,2,2,2,1,2 | 0.969 | 0.969 | 1.000 | 1.000 (0 outside) | 3 / 0 / 0 |
| 124 | trained | 5 | `0.result[0]:select,0.mlp_out:mlp,1.result[0]:select,1.mlp_out:mlp,2.mlp_out:mlp` | 1,1,2,2,2 | 0.963 | 0.984 | 0.800 | 1.000 (10 outside) | 3 / 0 / 0 |

Means: compiled: output agreement 0.901, counterfactual agreement 0.788, variables exact 13/18, bag heads 1; trained: output agreement 0.684, counterfactual agreement 0.546, variables exact 11/18, bag heads 0

Output agreement counts undefined positions as disagreement. A bag head is one whose attention pattern was not crisp enough for a selector, so its output is tabulated against the whole key multiset instead.

## Length generalization: InterpBench trained models (trained at nine tokens)

| Case | Tokens | Model token acc | Model seq acc | Program coverage | Program = model | Model acc where program defined | Model acc where program undefined |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4 | 0.031 | 0.000 | 0.000 | n/a | n/a | 0.031 |
| 2 | 5 | 0.196 | 0.000 | 0.000 | n/a | n/a | 0.196 |
| 2 | 6 | 0.173 | 0.000 | 0.000 | n/a | n/a | 0.173 |
| 2 | 7 | 0.109 | 0.000 | 0.000 | n/a | n/a | 0.109 |
| 2 | 8 | 0.062 | 0.000 | 0.000 | n/a | n/a | 0.062 |
| 2 | 9 | 1.000 | 1.000 | 0.000 | n/a | n/a | 1.000 |
| 7 | 4 | 0.575 | 0.135 | 0.909 | 0.692 | 0.532 | 1.000 |
| 7 | 5 | 0.628 | 0.040 | 0.928 | 0.628 | 0.599 | 1.000 |
| 7 | 6 | 0.643 | 0.040 | 0.966 | 0.682 | 0.634 | 0.902 |
| 7 | 7 | 0.628 | 0.100 | 0.618 | 0.948 | 0.948 | 0.110 |
| 7 | 8 | 0.845 | 0.550 | 0.645 | 0.968 | 0.972 | 0.614 |
| 7 | 9 | 1.000 | 1.000 | 0.980 | 0.760 | 1.000 | 1.000 |
| 13 | 4 | 0.940 | 0.825 | 0.497 | 1.000 | 1.000 | 0.881 |
| 13 | 5 | 0.993 | 0.965 | 0.582 | 1.000 | 1.000 | 0.983 |
| 13 | 6 | 0.981 | 0.890 | 0.660 | 1.000 | 1.000 | 0.944 |
| 13 | 7 | 0.977 | 0.840 | 0.704 | 1.000 | 1.000 | 0.923 |
| 13 | 8 | 0.974 | 0.815 | 0.742 | 1.000 | 1.000 | 0.901 |
| 13 | 9 | 1.000 | 1.000 | 0.960 | 0.995 | 1.000 | 1.000 |
| 21 | 4 | 1.000 | 1.000 | 0.875 | 0.891 | 1.000 | 1.000 |
| 21 | 5 | 1.000 | 1.000 | 0.836 | 0.842 | 1.000 | 1.000 |
| 21 | 6 | 1.000 | 1.000 | 0.823 | 0.845 | 1.000 | 1.000 |
| 21 | 7 | 1.000 | 1.000 | 0.825 | 0.804 | 1.000 | 1.000 |
| 21 | 8 | 1.000 | 1.000 | 0.797 | 0.795 | 1.000 | 1.000 |
| 21 | 9 | 1.000 | 1.000 | 0.741 | 0.761 | 1.000 | 1.000 |
| 44 | 4 | 0.901 | 0.665 | 0.752 | 0.925 | 1.000 | 0.601 |
| 44 | 5 | 0.923 | 0.680 | 0.863 | 0.906 | 0.997 | 0.460 |
| 44 | 6 | 0.897 | 0.520 | 0.880 | 0.899 | 1.000 | 0.146 |
| 44 | 7 | 0.893 | 0.430 | 0.877 | 0.888 | 0.999 | 0.134 |
| 44 | 8 | 0.902 | 0.420 | 0.889 | 0.899 | 0.999 | 0.124 |
| 44 | 9 | 1.000 | 1.000 | 0.892 | 0.914 | 1.000 | 1.000 |
| 124 | 4 | 0.990 | 0.990 | 0.708 | 0.548 | 0.993 | 0.983 |
| 124 | 5 | 0.995 | 0.995 | 0.711 | 0.592 | 0.993 | 1.000 |
| 124 | 6 | 0.993 | 0.985 | 0.718 | 0.601 | 0.991 | 1.000 |
| 124 | 7 | 0.990 | 0.990 | 0.706 | 0.667 | 0.986 | 1.000 |
| 124 | 8 | 1.000 | 1.000 | 0.764 | 0.640 | 1.000 | 1.000 |
| 124 | 9 | 1.000 | 1.000 | 0.735 | 0.680 | 1.000 | 1.000 |

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


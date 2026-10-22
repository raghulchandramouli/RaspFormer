# Circuit discovery: SIIT models

Edge and node scores use the absolute attribution score; node level promotes to sources by maximum, as in `bench.eap.evaluate_eap`. `ig` interpolates the embeddings; `ig-delta` adds alpha times the summed source difference to every target input. Means skip models without a defined metric.

| Model | Method | Edge AUROC | Edge AP | Edge P@k | Node AUROC | Node P@k |
| --- | --- | --- | --- | --- | --- | --- |
| P01_absolute | eap | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| P01_absolute | patch | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| P01_absolute | acdc | 0.843 | 0.127 | 0.000 | 0.800 | 0.000 |
| P01_absolute | acdc-logit | 0.800 | 0.108 | 0.000 | 0.600 | 0.000 |
| P01_absolute | ig | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| P01_absolute | ig-delta | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| P02_index_parity | eap | 0.564 | 0.231 | 0.333 | 0.333 | 0.333 |
| P02_index_parity | patch | 0.363 | 0.073 | 0.000 | 0.333 | 0.333 |
| P02_index_parity | acdc | 0.441 | 0.077 | 0.000 | 0.667 | 0.333 |
| P02_index_parity | acdc-logit | 0.412 | 0.076 | 0.000 | 0.333 | 0.333 |
| P02_index_parity | ig | 0.554 | 0.230 | 0.333 | 0.333 | 0.333 |
| P02_index_parity | ig-delta | 0.554 | 0.230 | 0.333 | 0.333 | 0.333 |
| P03_increment_by_index | eap | 0.730 | 0.694 | 0.667 | 0.667 | 0.667 |
| P03_increment_by_index | patch | 0.730 | 0.694 | 0.667 | 0.667 | 0.667 |
| P03_increment_by_index | acdc | 0.618 | 0.099 | 0.000 | 0.500 | 0.333 |
| P03_increment_by_index | acdc-logit | 0.603 | 0.100 | 0.000 | 0.500 | 0.333 |
| P03_increment_by_index | ig | 0.730 | 0.694 | 0.667 | 0.667 | 0.667 |
| P03_increment_by_index | ig-delta | 0.730 | 0.694 | 0.667 | 0.667 | 0.667 |
| P04_first_element | eap | 0.591 | 0.567 | 0.500 | 0.667 | 0.667 |
| P04_first_element | patch | 0.591 | 0.567 | 0.500 | 0.667 | 0.667 |
| P04_first_element | acdc | 0.682 | 0.451 | 0.500 | 0.583 | 0.667 |
| P04_first_element | acdc-logit | 0.561 | 0.484 | 0.500 | 0.333 | 0.333 |
| P04_first_element | ig | 0.591 | 0.567 | 0.500 | 0.667 | 0.667 |
| P04_first_element | ig-delta | 0.591 | 0.567 | 0.500 | 0.667 | 0.667 |
| P05_histogram | eap | 0.985 | 0.887 | 0.750 | 1.000 | 1.000 |
| P05_histogram | patch | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| P05_histogram | acdc | 0.985 | 0.887 | 0.750 | 1.000 | 1.000 |
| P05_histogram | acdc-logit | 0.985 | 0.887 | 0.750 | 1.000 | 1.000 |
| P05_histogram | ig | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| P05_histogram | ig-delta | 0.992 | 0.950 | 0.750 | 1.000 | 1.000 |
| P06_count_greater_than | eap | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| P06_count_greater_than | patch | 0.992 | 0.950 | 0.750 | 1.000 | 1.000 |
| P06_count_greater_than | acdc | 0.985 | 0.887 | 0.750 | 1.000 | 1.000 |
| P06_count_greater_than | acdc-logit | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| P06_count_greater_than | ig | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| P06_count_greater_than | ig-delta | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| P07_sum_with_next | eap | 0.699 | 0.385 | 0.455 | 0.671 | 0.571 |
| P07_sum_with_next | patch | 0.697 | 0.371 | 0.455 | 0.671 | 0.571 |
| P07_sum_with_next | acdc | 0.583 | 0.175 | 0.182 | 0.429 | 0.286 |
| P07_sum_with_next | acdc-logit | 0.493 | 0.135 | 0.091 | 0.429 | 0.286 |
| P07_sum_with_next | ig | 0.682 | 0.391 | 0.455 | 0.657 | 0.571 |
| P07_sum_with_next | ig-delta | 0.740 | 0.417 | 0.455 | 0.657 | 0.571 |
| P08_pairwise_sum | eap | 0.724 | 0.405 | 0.455 | 0.700 | 0.714 |
| P08_pairwise_sum | patch | 0.715 | 0.400 | 0.455 | 0.671 | 0.571 |
| P08_pairwise_sum | acdc | 0.563 | 0.173 | 0.182 | 0.429 | 0.286 |
| P08_pairwise_sum | acdc-logit | 0.560 | 0.048 | 0.000 | 0.429 | 0.286 |
| P08_pairwise_sum | ig | 0.716 | 0.406 | 0.455 | 0.686 | 0.714 |
| P08_pairwise_sum | ig-delta | 0.743 | 0.428 | 0.455 | 0.714 | 0.714 |
| P09_check_increasing | eap | 0.660 | 0.261 | 0.400 | 0.843 | 0.857 |
| P09_check_increasing | patch | 0.688 | 0.340 | 0.400 | 0.814 | 0.857 |
| P09_check_increasing | acdc | 0.771 | 0.143 | 0.200 | 0.686 | 0.714 |
| P09_check_increasing | acdc-logit | 0.687 | 0.158 | 0.300 | 0.793 | 0.714 |
| P09_check_increasing | ig | 0.705 | 0.360 | 0.500 | 0.800 | 0.714 |
| P09_check_increasing | ig-delta | 0.686 | 0.237 | 0.300 | 0.771 | 0.714 |
| P10_rotate_left | eap | 0.643 | 0.292 | 0.333 | 0.636 | 0.667 |
| P10_rotate_left | patch | 0.654 | 0.293 | 0.333 | 0.636 | 0.667 |
| P10_rotate_left | acdc | 0.698 | 0.220 | 0.222 | 0.455 | 0.333 |
| P10_rotate_left | acdc-logit | 0.619 | 0.070 | 0.111 | 0.576 | 0.500 |
| P10_rotate_left | ig | 0.598 | 0.284 | 0.222 | 0.636 | 0.667 |
| P10_rotate_left | ig-delta | 0.671 | 0.266 | 0.222 | 0.652 | 0.667 |
| P11_check_palindrome | eap | 0.740 | 0.154 | 0.125 | 0.827 | 0.727 |
| P11_check_palindrome | patch | 0.752 | 0.113 | 0.188 | 0.831 | 0.727 |
| P11_check_palindrome | acdc | 0.757 | 0.132 | 0.125 | 0.578 | 0.273 |
| P11_check_palindrome | acdc-logit | 0.739 | 0.082 | 0.062 | 0.619 | 0.273 |
| P11_check_palindrome | ig | 0.759 | 0.120 | 0.125 | 0.805 | 0.636 |
| P11_check_palindrome | ig-delta | 0.682 | 0.131 | 0.188 | 0.827 | 0.727 |
| P12_sorting | eap | 0.754 | 0.397 | 0.333 | 0.818 | 0.833 |
| P12_sorting | patch | 0.791 | 0.755 | 0.778 | 0.833 | 0.833 |
| P12_sorting | acdc | 0.702 | 0.062 | 0.000 | 0.530 | 0.333 |
| P12_sorting | acdc-logit | 0.758 | 0.111 | 0.000 | 0.492 | 0.333 |
| P12_sorting | ig | 0.784 | 0.657 | 0.667 | 0.833 | 0.833 |
| P12_sorting | ig-delta | 0.765 | 0.398 | 0.444 | 0.788 | 0.833 |
| **mean** | eap | 0.758 | 0.523 | 0.529 | 0.764 | 0.753 |
| **mean** | patch | 0.748 | 0.546 | 0.544 | 0.760 | 0.741 |
| **mean** | acdc | 0.719 | 0.286 | 0.243 | 0.638 | 0.463 |
| **mean** | acdc-logit | 0.685 | 0.272 | 0.235 | 0.592 | 0.449 |
| **mean** | ig | 0.760 | 0.559 | 0.577 | 0.757 | 0.734 |
| **mean** | ig-delta | 0.763 | 0.527 | 0.526 | 0.756 | 0.741 |

## ACDC circuits by threshold

Precision, recall and F1 of the kept edges (and of the sources with a kept edge) against the true circuit.

| Model | Method | tau | Edges kept | Edge P | Edge R | Edge F1 | Nodes kept | Node P | Node R | Node F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P01_absolute | acdc | 0.0001 | 13 | 0.154 | 1.000 | 0.267 | 6 | 0.333 | 1.000 | 0.500 |
| P01_absolute | acdc | 0.001 | 14 | 0.143 | 1.000 | 0.250 | 5 | 0.400 | 1.000 | 0.571 |
| P01_absolute | acdc | 0.01 | 16 | 0.125 | 1.000 | 0.222 | 4 | 0.500 | 1.000 | 0.667 |
| P01_absolute | acdc | 0.1 | 13 | 0.154 | 1.000 | 0.267 | 4 | 0.500 | 1.000 | 0.667 |
| P01_absolute | acdc-logit | 0.0001 | 23 | 0.087 | 1.000 | 0.160 | 6 | 0.333 | 1.000 | 0.500 |
| P01_absolute | acdc-logit | 0.001 | 23 | 0.087 | 1.000 | 0.160 | 6 | 0.333 | 1.000 | 0.500 |
| P01_absolute | acdc-logit | 0.01 | 22 | 0.091 | 1.000 | 0.167 | 6 | 0.333 | 1.000 | 0.500 |
| P01_absolute | acdc-logit | 0.1 | 16 | 0.125 | 1.000 | 0.222 | 6 | 0.333 | 1.000 | 0.500 |
| P02_index_parity | acdc | 0.0001 | 4 | 0.000 | 0.000 | 0.000 | 1 | 1.000 | 0.333 | 0.500 |
| P02_index_parity | acdc | 0.001 | 0 | 0.000 | 0.000 | 0.000 | 0 | 0.000 | 0.000 | 0.000 |
| P02_index_parity | acdc | 0.01 | 0 | 0.000 | 0.000 | 0.000 | 0 | 0.000 | 0.000 | 0.000 |
| P02_index_parity | acdc | 0.1 | 0 | 0.000 | 0.000 | 0.000 | 0 | 0.000 | 0.000 | 0.000 |
| P02_index_parity | acdc-logit | 0.0001 | 16 | 0.062 | 0.333 | 0.105 | 5 | 0.200 | 0.333 | 0.250 |
| P02_index_parity | acdc-logit | 0.001 | 15 | 0.067 | 0.333 | 0.111 | 4 | 0.250 | 0.333 | 0.286 |
| P02_index_parity | acdc-logit | 0.01 | 8 | 0.000 | 0.000 | 0.000 | 1 | 1.000 | 0.333 | 0.500 |
| P02_index_parity | acdc-logit | 0.1 | 1 | 0.000 | 0.000 | 0.000 | 1 | 1.000 | 0.333 | 0.500 |
| P03_increment_by_index | acdc | 0.0001 | 17 | 0.118 | 0.667 | 0.200 | 6 | 0.333 | 0.667 | 0.444 |
| P03_increment_by_index | acdc | 0.001 | 17 | 0.118 | 0.667 | 0.200 | 5 | 0.400 | 0.667 | 0.500 |
| P03_increment_by_index | acdc | 0.01 | 16 | 0.125 | 0.667 | 0.211 | 5 | 0.400 | 0.667 | 0.500 |
| P03_increment_by_index | acdc | 0.1 | 16 | 0.125 | 0.667 | 0.211 | 4 | 0.500 | 0.667 | 0.571 |
| P03_increment_by_index | acdc-logit | 0.0001 | 23 | 0.087 | 0.667 | 0.154 | 6 | 0.333 | 0.667 | 0.444 |
| P03_increment_by_index | acdc-logit | 0.001 | 23 | 0.087 | 0.667 | 0.154 | 6 | 0.333 | 0.667 | 0.444 |
| P03_increment_by_index | acdc-logit | 0.01 | 19 | 0.105 | 0.667 | 0.182 | 6 | 0.333 | 0.667 | 0.444 |
| P03_increment_by_index | acdc-logit | 0.1 | 15 | 0.133 | 0.667 | 0.222 | 4 | 0.500 | 0.667 | 0.571 |
| P04_first_element | acdc | 0.0001 | 7 | 0.286 | 0.500 | 0.364 | 5 | 0.400 | 0.667 | 0.500 |
| P04_first_element | acdc | 0.001 | 7 | 0.286 | 0.500 | 0.364 | 5 | 0.400 | 0.667 | 0.500 |
| P04_first_element | acdc | 0.01 | 6 | 0.333 | 0.500 | 0.400 | 4 | 0.500 | 0.667 | 0.571 |
| P04_first_element | acdc | 0.1 | 5 | 0.400 | 0.500 | 0.444 | 3 | 0.667 | 0.667 | 0.667 |
| P04_first_element | acdc-logit | 0.0001 | 21 | 0.095 | 0.500 | 0.160 | 6 | 0.333 | 0.667 | 0.444 |
| P04_first_element | acdc-logit | 0.001 | 18 | 0.111 | 0.500 | 0.182 | 6 | 0.333 | 0.667 | 0.444 |
| P04_first_element | acdc-logit | 0.01 | 11 | 0.182 | 0.500 | 0.267 | 6 | 0.333 | 0.667 | 0.444 |
| P04_first_element | acdc-logit | 0.1 | 8 | 0.250 | 0.500 | 0.333 | 6 | 0.333 | 0.667 | 0.444 |
| P05_histogram | acdc | 0.0001 | 21 | 0.190 | 1.000 | 0.320 | 6 | 0.500 | 1.000 | 0.667 |
| P05_histogram | acdc | 0.001 | 18 | 0.222 | 1.000 | 0.364 | 6 | 0.500 | 1.000 | 0.667 |
| P05_histogram | acdc | 0.01 | 11 | 0.364 | 1.000 | 0.533 | 4 | 0.750 | 1.000 | 0.857 |
| P05_histogram | acdc | 0.1 | 5 | 0.800 | 1.000 | 0.889 | 3 | 1.000 | 1.000 | 1.000 |
| P05_histogram | acdc-logit | 0.0001 | 22 | 0.182 | 1.000 | 0.308 | 6 | 0.500 | 1.000 | 0.667 |
| P05_histogram | acdc-logit | 0.001 | 18 | 0.222 | 1.000 | 0.364 | 6 | 0.500 | 1.000 | 0.667 |
| P05_histogram | acdc-logit | 0.01 | 13 | 0.308 | 1.000 | 0.471 | 4 | 0.750 | 1.000 | 0.857 |
| P05_histogram | acdc-logit | 0.1 | 5 | 0.800 | 1.000 | 0.889 | 3 | 1.000 | 1.000 | 1.000 |
| P06_count_greater_than | acdc | 0.0001 | 17 | 0.235 | 1.000 | 0.381 | 5 | 0.600 | 1.000 | 0.750 |
| P06_count_greater_than | acdc | 0.001 | 15 | 0.267 | 1.000 | 0.421 | 4 | 0.750 | 1.000 | 0.857 |
| P06_count_greater_than | acdc | 0.01 | 15 | 0.267 | 1.000 | 0.421 | 3 | 1.000 | 1.000 | 1.000 |
| P06_count_greater_than | acdc | 0.1 | 5 | 0.800 | 1.000 | 0.889 | 3 | 1.000 | 1.000 | 1.000 |
| P06_count_greater_than | acdc-logit | 0.0001 | 22 | 0.182 | 1.000 | 0.308 | 6 | 0.500 | 1.000 | 0.667 |
| P06_count_greater_than | acdc-logit | 0.001 | 20 | 0.200 | 1.000 | 0.333 | 6 | 0.500 | 1.000 | 0.667 |
| P06_count_greater_than | acdc-logit | 0.01 | 14 | 0.286 | 1.000 | 0.444 | 5 | 0.600 | 1.000 | 0.750 |
| P06_count_greater_than | acdc-logit | 0.1 | 4 | 1.000 | 1.000 | 1.000 | 3 | 1.000 | 1.000 | 1.000 |
| P07_sum_with_next | acdc | 0.0001 | 183 | 0.044 | 0.727 | 0.082 | 16 | 0.375 | 0.857 | 0.522 |
| P07_sum_with_next | acdc | 0.001 | 161 | 0.050 | 0.727 | 0.093 | 16 | 0.375 | 0.857 | 0.522 |
| P07_sum_with_next | acdc | 0.01 | 143 | 0.063 | 0.818 | 0.117 | 16 | 0.375 | 0.857 | 0.522 |
| P07_sum_with_next | acdc | 0.1 | 141 | 0.043 | 0.545 | 0.079 | 16 | 0.375 | 0.857 | 0.522 |
| P07_sum_with_next | acdc-logit | 0.0001 | 222 | 0.041 | 0.818 | 0.077 | 16 | 0.375 | 0.857 | 0.522 |
| P07_sum_with_next | acdc-logit | 0.001 | 199 | 0.040 | 0.727 | 0.076 | 16 | 0.375 | 0.857 | 0.522 |
| P07_sum_with_next | acdc-logit | 0.01 | 184 | 0.033 | 0.545 | 0.062 | 16 | 0.375 | 0.857 | 0.522 |
| P07_sum_with_next | acdc-logit | 0.1 | 134 | 0.037 | 0.455 | 0.069 | 16 | 0.375 | 0.857 | 0.522 |
| P08_pairwise_sum | acdc | 0.0001 | 192 | 0.042 | 0.727 | 0.079 | 16 | 0.375 | 0.857 | 0.522 |
| P08_pairwise_sum | acdc | 0.001 | 180 | 0.044 | 0.727 | 0.084 | 16 | 0.375 | 0.857 | 0.522 |
| P08_pairwise_sum | acdc | 0.01 | 156 | 0.051 | 0.727 | 0.096 | 16 | 0.375 | 0.857 | 0.522 |
| P08_pairwise_sum | acdc | 0.1 | 143 | 0.042 | 0.545 | 0.078 | 16 | 0.375 | 0.857 | 0.522 |
| P08_pairwise_sum | acdc-logit | 0.0001 | 241 | 0.037 | 0.818 | 0.071 | 16 | 0.375 | 0.857 | 0.522 |
| P08_pairwise_sum | acdc-logit | 0.001 | 197 | 0.046 | 0.818 | 0.087 | 16 | 0.375 | 0.857 | 0.522 |
| P08_pairwise_sum | acdc-logit | 0.01 | 168 | 0.042 | 0.636 | 0.078 | 16 | 0.375 | 0.857 | 0.522 |
| P08_pairwise_sum | acdc-logit | 0.1 | 118 | 0.042 | 0.455 | 0.078 | 16 | 0.375 | 0.857 | 0.522 |
| P09_check_increasing | acdc | 0.0001 | 34 | 0.206 | 0.700 | 0.318 | 11 | 0.545 | 0.857 | 0.667 |
| P09_check_increasing | acdc | 0.001 | 39 | 0.154 | 0.600 | 0.245 | 12 | 0.417 | 0.714 | 0.526 |
| P09_check_increasing | acdc | 0.01 | 39 | 0.154 | 0.600 | 0.245 | 12 | 0.417 | 0.714 | 0.526 |
| P09_check_increasing | acdc | 0.1 | 32 | 0.156 | 0.500 | 0.238 | 7 | 0.714 | 0.714 | 0.714 |
| P09_check_increasing | acdc-logit | 0.0001 | 225 | 0.031 | 0.700 | 0.060 | 16 | 0.375 | 0.857 | 0.522 |
| P09_check_increasing | acdc-logit | 0.001 | 118 | 0.059 | 0.700 | 0.109 | 15 | 0.400 | 0.857 | 0.545 |
| P09_check_increasing | acdc-logit | 0.01 | 41 | 0.146 | 0.600 | 0.235 | 9 | 0.667 | 0.857 | 0.750 |
| P09_check_increasing | acdc-logit | 0.1 | 8 | 0.375 | 0.300 | 0.333 | 3 | 1.000 | 0.429 | 0.600 |
| P10_rotate_left | acdc | 0.0001 | 77 | 0.065 | 0.556 | 0.116 | 14 | 0.357 | 0.833 | 0.500 |
| P10_rotate_left | acdc | 0.001 | 57 | 0.070 | 0.444 | 0.121 | 13 | 0.385 | 0.833 | 0.526 |
| P10_rotate_left | acdc | 0.01 | 38 | 0.105 | 0.444 | 0.170 | 9 | 0.444 | 0.667 | 0.533 |
| P10_rotate_left | acdc | 0.1 | 66 | 0.076 | 0.556 | 0.133 | 15 | 0.333 | 0.833 | 0.476 |
| P10_rotate_left | acdc-logit | 0.0001 | 155 | 0.039 | 0.667 | 0.073 | 16 | 0.312 | 0.833 | 0.455 |
| P10_rotate_left | acdc-logit | 0.001 | 87 | 0.046 | 0.444 | 0.083 | 16 | 0.312 | 0.833 | 0.455 |
| P10_rotate_left | acdc-logit | 0.01 | 40 | 0.075 | 0.333 | 0.122 | 14 | 0.357 | 0.833 | 0.500 |
| P10_rotate_left | acdc-logit | 0.1 | 20 | 0.100 | 0.222 | 0.138 | 9 | 0.444 | 0.667 | 0.533 |
| P11_check_palindrome | acdc | 0.0001 | 313 | 0.038 | 0.750 | 0.073 | 26 | 0.346 | 0.818 | 0.486 |
| P11_check_palindrome | acdc | 0.001 | 252 | 0.040 | 0.625 | 0.075 | 24 | 0.375 | 0.818 | 0.514 |
| P11_check_palindrome | acdc | 0.01 | 228 | 0.044 | 0.625 | 0.082 | 25 | 0.360 | 0.818 | 0.500 |
| P11_check_palindrome | acdc | 0.1 | 210 | 0.052 | 0.688 | 0.097 | 22 | 0.409 | 0.818 | 0.545 |
| P11_check_palindrome | acdc-logit | 0.0001 | 760 | 0.017 | 0.812 | 0.034 | 30 | 0.333 | 0.909 | 0.488 |
| P11_check_palindrome | acdc-logit | 0.001 | 440 | 0.030 | 0.812 | 0.057 | 27 | 0.370 | 0.909 | 0.526 |
| P11_check_palindrome | acdc-logit | 0.01 | 219 | 0.041 | 0.562 | 0.077 | 23 | 0.391 | 0.818 | 0.529 |
| P11_check_palindrome | acdc-logit | 0.1 | 39 | 0.077 | 0.188 | 0.109 | 18 | 0.444 | 0.727 | 0.552 |
| P12_sorting | acdc | 0.0001 | 172 | 0.041 | 0.778 | 0.077 | 16 | 0.312 | 0.833 | 0.455 |
| P12_sorting | acdc | 0.001 | 116 | 0.060 | 0.778 | 0.112 | 15 | 0.333 | 0.833 | 0.476 |
| P12_sorting | acdc | 0.01 | 108 | 0.065 | 0.778 | 0.120 | 13 | 0.385 | 0.833 | 0.526 |
| P12_sorting | acdc | 0.1 | 100 | 0.070 | 0.778 | 0.128 | 13 | 0.385 | 0.833 | 0.526 |
| P12_sorting | acdc-logit | 0.0001 | 213 | 0.033 | 0.778 | 0.063 | 16 | 0.312 | 0.833 | 0.455 |
| P12_sorting | acdc-logit | 0.001 | 145 | 0.048 | 0.778 | 0.091 | 16 | 0.312 | 0.833 | 0.455 |
| P12_sorting | acdc-logit | 0.01 | 87 | 0.080 | 0.778 | 0.146 | 16 | 0.312 | 0.833 | 0.455 |
| P12_sorting | acdc-logit | 0.1 | 45 | 0.156 | 0.778 | 0.259 | 14 | 0.357 | 0.833 | 0.500 |

Means by method and tau: acdc tau 0.0001: edge F1 0.190, node F1 0.543; acdc tau 0.001: edge F1 0.194, node F1 0.515; acdc tau 0.01: edge F1 0.218, node F1 0.560; acdc tau 0.1: edge F1 0.288, node F1 0.601; acdc-logit tau 0.0001: edge F1 0.131, node F1 0.495; acdc-logit tau 0.001: edge F1 0.151, node F1 0.503; acdc-logit tau 0.01: edge F1 0.188, node F1 0.564; acdc-logit tau 0.1: edge F1 0.304, node F1 0.604

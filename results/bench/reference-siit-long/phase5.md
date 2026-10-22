# Phase 5: length generalization beyond the training length

SIIT models trained on six-token sequences with ten-token position embeddings, tested from three to ten tokens against the compiled reference.

## Mean token accuracy over the twelve programs

| Tokens | Model token acc | Model seq acc | Program coverage |
| --- | --- | --- | --- |
| 3 | 0.858 | 0.729 | 0.698 |
| 4 | 0.872 | 0.652 | 0.706 |
| 5 | 0.908 | 0.669 | 0.700 |
| 6 | 1.000 | 1.000 | 0.704 |
| 7 | 0.843 | 0.446 | 0.434 |
| 8 | 0.765 | 0.350 | 0.410 |
| 9 | 0.645 | 0.253 | 0.390 |
| 10 | 0.614 | 0.251 | 0.375 |

## Per program

| Program | Tokens | Model token acc | Model seq acc | Program coverage | Program = model | Model acc where defined | Model acc where undefined |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P01_absolute | 3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | n/a |
| P01_absolute | 4 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | n/a |
| P01_absolute | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | n/a |
| P01_absolute | 6 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | n/a |
| P01_absolute | 7 (longer) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | n/a |
| P01_absolute | 8 (longer) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | n/a |
| P01_absolute | 9 (longer) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | n/a |
| P01_absolute | 10 (longer) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | n/a |
| P02_index_parity | 3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | n/a |
| P02_index_parity | 4 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | n/a |
| P02_index_parity | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | n/a |
| P02_index_parity | 6 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | n/a |
| P02_index_parity | 7 (longer) | 1.000 | 1.000 | 0.857 | 1.000 | 1.000 | 1.000 |
| P02_index_parity | 8 (longer) | 0.881 | 0.047 | 0.750 | 1.000 | 1.000 | 0.523 |
| P02_index_parity | 9 (longer) | 0.890 | 0.007 | 0.667 | 1.000 | 1.000 | 0.669 |
| P02_index_parity | 10 (longer) | 0.830 | 0.000 | 0.600 | 1.000 | 1.000 | 0.576 |
| P03_increment_by_index | 3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | n/a |
| P03_increment_by_index | 4 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | n/a |
| P03_increment_by_index | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | n/a |
| P03_increment_by_index | 6 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | n/a |
| P03_increment_by_index | 7 (longer) | 0.857 | 0.000 | 0.857 | 1.000 | 1.000 | 0.000 |
| P03_increment_by_index | 8 (longer) | 0.750 | 0.000 | 0.750 | 1.000 | 1.000 | 0.000 |
| P03_increment_by_index | 9 (longer) | 0.667 | 0.000 | 0.667 | 1.000 | 1.000 | 0.000 |
| P03_increment_by_index | 10 (longer) | 0.600 | 0.000 | 0.600 | 1.000 | 1.000 | 0.000 |
| P04_first_element | 3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | n/a |
| P04_first_element | 4 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | n/a |
| P04_first_element | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | n/a |
| P04_first_element | 6 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | n/a |
| P04_first_element | 7 (longer) | 1.000 | 1.000 | 0.000 | n/a | n/a | 1.000 |
| P04_first_element | 8 (longer) | 1.000 | 1.000 | 0.000 | n/a | n/a | 1.000 |
| P04_first_element | 9 (longer) | 1.000 | 1.000 | 0.000 | n/a | n/a | 1.000 |
| P04_first_element | 10 (longer) | 1.000 | 1.000 | 0.000 | n/a | n/a | 1.000 |
| P05_histogram | 3 | 0.633 | 0.273 | 0.587 | 0.633 | 0.633 | 0.634 |
| P05_histogram | 4 | 0.732 | 0.297 | 0.512 | 0.828 | 0.732 | 0.732 |
| P05_histogram | 5 | 0.721 | 0.210 | 0.426 | 0.859 | 0.721 | 0.720 |
| P05_histogram | 6 | 1.000 | 1.000 | 0.335 | 0.836 | 1.000 | 1.000 |
| P05_histogram | 7 (longer) | 0.809 | 0.287 | 0.000 | n/a | n/a | 0.809 |
| P05_histogram | 8 (longer) | 0.712 | 0.140 | 0.000 | n/a | n/a | 0.712 |
| P05_histogram | 9 (longer) | 0.665 | 0.030 | 0.000 | n/a | n/a | 0.665 |
| P05_histogram | 10 (longer) | 0.608 | 0.007 | 0.000 | n/a | n/a | 0.608 |
| P06_count_greater_than | 3 | 0.868 | 0.667 | 0.452 | 0.725 | 0.747 | 0.968 |
| P06_count_greater_than | 4 | 0.833 | 0.383 | 0.546 | 0.841 | 0.803 | 0.870 |
| P06_count_greater_than | 5 | 0.897 | 0.567 | 0.609 | 0.908 | 0.831 | 1.000 |
| P06_count_greater_than | 6 | 1.000 | 1.000 | 0.599 | 0.964 | 1.000 | 1.000 |
| P06_count_greater_than | 7 (longer) | 0.685 | 0.020 | 0.498 | 0.925 | 0.781 | 0.590 |
| P06_count_greater_than | 8 (longer) | 0.543 | 0.000 | 0.420 | 0.857 | 0.757 | 0.388 |
| P06_count_greater_than | 9 (longer) | 0.450 | 0.000 | 0.343 | 0.744 | 0.712 | 0.313 |
| P06_count_greater_than | 10 (longer) | 0.401 | 0.000 | 0.295 | 0.675 | 0.674 | 0.287 |
| P07_sum_with_next | 3 | 1.000 | 1.000 | 0.306 | 0.880 | 1.000 | 1.000 |
| P07_sum_with_next | 4 | 1.000 | 1.000 | 0.301 | 0.751 | 1.000 | 1.000 |
| P07_sum_with_next | 5 | 1.000 | 1.000 | 0.246 | 0.770 | 1.000 | 1.000 |
| P07_sum_with_next | 6 | 1.000 | 1.000 | 0.199 | 0.821 | 1.000 | 1.000 |
| P07_sum_with_next | 7 (longer) | 0.743 | 0.013 | 0.000 | n/a | n/a | 0.743 |
| P07_sum_with_next | 8 (longer) | 0.662 | 0.003 | 0.000 | n/a | n/a | 0.662 |
| P07_sum_with_next | 9 (longer) | 0.600 | 0.000 | 0.000 | n/a | n/a | 0.600 |
| P07_sum_with_next | 10 (longer) | 0.556 | 0.003 | 0.000 | n/a | n/a | 0.556 |
| P08_pairwise_sum | 3 | 0.714 | 0.143 | 0.368 | 0.861 | 1.000 | 0.548 |
| P08_pairwise_sum | 4 | 0.778 | 0.113 | 0.362 | 0.830 | 1.000 | 0.652 |
| P08_pairwise_sum | 5 | 0.827 | 0.137 | 0.322 | 0.787 | 1.000 | 0.745 |
| P08_pairwise_sum | 6 | 1.000 | 1.000 | 0.314 | 0.754 | 1.000 | 1.000 |
| P08_pairwise_sum | 7 (longer) | 0.765 | 0.020 | 0.000 | n/a | n/a | 0.765 |
| P08_pairwise_sum | 8 (longer) | 0.685 | 0.013 | 0.000 | n/a | n/a | 0.685 |
| P08_pairwise_sum | 9 (longer) | 0.627 | 0.000 | 0.000 | n/a | n/a | 0.627 |
| P08_pairwise_sum | 10 (longer) | 0.587 | 0.000 | 0.000 | n/a | n/a | 0.587 |
| P09_check_increasing | 3 | 0.832 | 0.823 | 1.000 | 0.897 | 0.832 | n/a |
| P09_check_increasing | 4 | 0.948 | 0.947 | 1.000 | 0.965 | 0.948 | n/a |
| P09_check_increasing | 5 | 0.991 | 0.990 | 1.000 | 0.994 | 0.991 | n/a |
| P09_check_increasing | 6 | 1.000 | 1.000 | 1.000 | 0.997 | 1.000 | n/a |
| P09_check_increasing | 7 (longer) | 0.995 | 0.993 | 1.000 | 0.995 | 0.995 | n/a |
| P09_check_increasing | 8 (longer) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | n/a |
| P09_check_increasing | 9 (longer) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | n/a |
| P09_check_increasing | 10 (longer) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | n/a |
| P10_rotate_left | 3 | 0.986 | 0.957 | 0.667 | 0.687 | 1.000 | 0.957 |
| P10_rotate_left | 4 | 0.777 | 0.110 | 0.750 | 0.630 | 1.000 | 0.110 |
| P10_rotate_left | 5 | 0.826 | 0.130 | 0.800 | 0.660 | 1.000 | 0.130 |
| P10_rotate_left | 6 | 1.000 | 1.000 | 1.000 | 0.673 | 1.000 | n/a |
| P10_rotate_left | 7 (longer) | 0.766 | 0.013 | 0.000 | n/a | n/a | 0.766 |
| P10_rotate_left | 8 (longer) | 0.670 | 0.000 | 0.000 | n/a | n/a | 0.670 |
| P10_rotate_left | 9 (longer) | 0.617 | 0.000 | 0.000 | n/a | n/a | 0.617 |
| P10_rotate_left | 10 (longer) | 0.571 | 0.000 | 0.000 | n/a | n/a | 0.571 |
| P11_check_palindrome | 3 | 0.852 | 0.850 | 1.000 | 0.962 | 0.852 | n/a |
| P11_check_palindrome | 4 | 0.980 | 0.980 | 1.000 | 0.993 | 0.980 | n/a |
| P11_check_palindrome | 5 | 0.987 | 0.987 | 1.000 | 0.997 | 0.987 | n/a |
| P11_check_palindrome | 6 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | n/a |
| P11_check_palindrome | 7 (longer) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | n/a |
| P11_check_palindrome | 8 (longer) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | n/a |
| P11_check_palindrome | 9 (longer) | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | n/a |
| P11_check_palindrome | 10 (longer) | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 | n/a |
| P12_sorting | 3 | 0.406 | 0.030 | 0.000 | n/a | n/a | 0.406 |
| P12_sorting | 4 | 0.420 | 0.000 | 0.000 | n/a | n/a | 0.420 |
| P12_sorting | 5 | 0.649 | 0.007 | 0.000 | n/a | n/a | 0.649 |
| P12_sorting | 6 | 1.000 | 1.000 | 0.000 | n/a | n/a | 1.000 |
| P12_sorting | 7 (longer) | 0.493 | 0.007 | 0.000 | n/a | n/a | 0.493 |
| P12_sorting | 8 (longer) | 0.279 | 0.000 | 0.000 | n/a | n/a | 0.279 |
| P12_sorting | 9 (longer) | 0.223 | 0.000 | 0.000 | n/a | n/a | 0.223 |
| P12_sorting | 10 (longer) | 0.219 | 0.000 | 0.000 | n/a | n/a | 0.219 |

Longer than training (48 program-length rows): mean model token accuracy 0.717, mean program coverage 0.402.
Where the recovered program is defined the model scores 0.910 on average; where it is undefined 0.362 (12 rows with both).

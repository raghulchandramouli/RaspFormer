# Tracr ground-truth schedules

Tracr 1.0.0 (`9ce2b8c82b6ba10e62e86cf6f390e7536d4fd2cd`); 12/12 compiled; 60 sampled outputs matched RASP.

## P01_absolute

| Variable | Lanes | First appears |
|---|---:|---|
| tokens | 12–19 | input embedding |
| absolute_1 | 0–5 | after mlp1 |


## P02_index_parity

| Variable | Lanes | First appears |
|---|---:|---|
| indices | 2–7 | input embedding |
| tokens | 9–16 | input embedding |
| index_parity_2 | 0–1 | after mlp1 |


## P03_increment_by_index

| Variable | Lanes | First appears |
|---|---:|---|
| indices | 13–18 | input embedding |
| tokens | 20–27 | input embedding |
| increment_by_index_3 | 0–12 | after mlp1 |


## P04_first_element

| Variable | Lanes | First appears |
|---|---:|---|
| indices | 8–13 | input embedding |
| tokens | 15–22 | input embedding |
| first_element_4 | 0–7 | after attention1 |


## P05_histogram

| Variable | Lanes | First appears |
|---|---:|---|
| tokens | 15–22 | input embedding |
| histogram_6 | 0–6 | after mlp1 |


## P06_count_greater_than

| Variable | Lanes | First appears |
|---|---:|---|
| tokens | 15–22 | input embedding |
| count_greater_than_8 | 0–6 | after mlp1 |


## P07_sum_with_next

| Variable | Lanes | First appears |
|---|---:|---|
| indices | 0–5 | input embedding |
| tokens | 45–52 | input embedding |
| length_14 | 6–12 | after mlp1 |
| next_clamped_index_13 | 14–20 | after mlp2 |
| next_value_11 | 21–28 | after attention3 |
| sum_with_next_10 | 30–44 | after mlp3 |


## P08_pairwise_sum

| Variable | Lanes | First appears |
|---|---:|---|
| indices | 0–5 | input embedding |
| tokens | 44–51 | input embedding |
| length_20 | 6–12 | after mlp1 |
| previous_cyclic_index_19 | 30–35 | after mlp2 |
| previous_value_17 | 36–43 | after attention3 |
| pairwise_sum_16 | 15–29 | after mlp3 |


## P09_check_increasing

| Variable | Lanes | First appears |
|---|---:|---|
| indices | 10–15 | input embedding |
| tokens | 27–34 | input embedding |
| shift_by(1)_26 | 19–26 | after attention1 |
| locally_increasing_25 | 16–17 | after mlp1 |
| decrease_count_23 | 2–8 | after mlp2 |
| check_increasing_22 | 0–1 | after mlp3 |


## P10_rotate_left

| Variable | Lanes | First appears |
|---|---:|---|
| indices | 0–5 | input embedding |
| tokens | 29–36 | input embedding |
| length_31 | 6–12 | after mlp1 |
| next_index_30 | 14–19 | after mlp2 |
| rotate_left_28 | 21–28 | after attention3 |


## P11_check_palindrome

| Variable | Lanes | First appears |
|---|---:|---|
| indices | 2–7 | input embedding |
| tokens | 59–66 | input embedding |
| length_41 | 8–14 | after mlp1 |
| opp_idx_40 | 39–50 | after mlp2 |
| opp_idx-1_39 | 27–38 | after mlp3 |
| reverse_37 | 51–58 | after attention4 |
| mirror_matches_36 | 16–17 | after mlp4 |
| mismatch_count_34 | 18–24 | after mlp5 |
| check_palindrome_33 | 0–1 | after mlp6 |


## P12_sorting

| Variable | Lanes | First appears |
|---|---:|---|
| indices | 0–5 | input embedding |
| tokens | 71–78 | input embedding |
| sequence_map_47 | 7–54 | after mlp1 |
| target_pos_45 | 63–69 | after mlp2 |
| sorting_43 | 55–62 | after attention3 |

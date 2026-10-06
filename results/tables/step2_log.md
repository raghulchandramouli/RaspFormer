# Step 2: Manual forward-pass trace summary

Source: [step2_manual_trace.json](step2_manual_trace.json). The trace code and saved execution output are in [compile.ipynb](../../compile.ipynb).

## What this experiment does

For each RASP program, we run the same input through three routes:

1. **The original RASP program** produces the expected answer.
2. **The compiled Tracr Transformer** produces output scores and a decoded answer.
3. **Our manual forward pass** rebuilds the input embeddings, follows the compiled calculation blocks, and produces output scores.

We compare the manual scores with the Transformer's scores, then compare the Transformer's decoded answer with the RASP answer. This checks that the traced calculations reproduce the compiled model's output and that the model implements the intended program.

## Overall result

**All 70 saved cases pass the final-score and decoded-answer checks.**

| Measurement | Result |
| --- | --- |
| Programs | 14: the original 12, plus focused programs A and B |
| Inputs per program | 5 |
| Manual/model final-score comparisons passing | 70 / 70 cases |
| Model/RASP decoded answers matching | 70 / 70 cases; 252 token positions |
| Individual output-score values compared | 2,369, including the beginning-of-sequence row |
| Manual intermediate block snapshots | 220 |
| Largest absolute final-score difference | `1.9073486328125e-6` (about `0.00000191`) |
| Notebook comparison tolerances | Absolute `1e-4`; relative `1e-4` |

The largest score difference occurs in `P05_histogram` and `P06_count_greater_than`. Every saved score satisfies the notebook's elementwise tolerance rule:

```text
abs(manual_score - model_score) <= 1e-4 + 1e-4 * abs(model_score)
```

The saved decoded answers match the RASP answers exactly for these inputs. All 70 stored `max_abs_score_error` values also agree with differences recalculated from the saved arrays.

## What happens during the manual pass

1. Add a beginning-of-sequence token (`BOS`), then construct token and position embeddings.
2. Run each compiled attention or MLP block in order. Attention computes scaled scores, applies softmax, and combines values. The manual routine uses the assembled model's `key_size` for the scaling.
3. Add each block's contribution to the **residual stream**, the model's running table of values for every position.
4. Save the updated residual and its output-channel scores after each block.
5. Read the final output channels and compare them with the real model. For categorical outputs, these are the scores before selecting the winning category.
6. Compare the model's decoded answer with the original RASP program's answer.

Score and residual arrays include the `BOS` row. The saved `rasp_output` and `model_decoded` lists contain only the actual input positions.

## Results by program

**Path notation:** `A1` means attention in Transformer layer 1; `M1` means the MLP in layer 1. The JSON uses zero-based names, so `transformer/layer_0/attn` becomes `A1`. Paths list the compiled blocks recorded in `block_trace`. Every program passes all five final-score and decoded-answer checks.

| Program | What it computes | Recorded block path | Worst final-score difference |
| --- | --- | --- | --- |
| `P01_absolute` | Absolute value of each token | M1 | `0` |
| `P02_index_parity` | Whether each zero-based position is even or odd | M1 | `0` |
| `P03_increment_by_index` | Each token plus its position index | M1 | `0` |
| `P04_first_element` | Repeat the first token at every position | A1 | `2.10e-8` |
| `P05_histogram` | How often each token occurs in the sequence | A1 → M1 | `1.91e-6` |
| `P06_count_greater_than` | How many tokens are strictly greater than each token | A1 → M1 | `1.91e-6` |
| `P07_sum_with_next` | Each token plus the next; the last pairs with itself | A1 → M1 → M2 → A3 → M3 | `1.67e-8` |
| `P08_pairwise_sum` | Each token plus the previous; the first pairs with the last | A1 → M1 → M2 → A3 → M3 | `1.67e-8` |
| `P09_check_increasing` | Repeat a flag indicating whether the sequence is nondecreasing | A1 → M1 → A2 → M2 → M3 | `0` |
| `P10_rotate_left` | Shift the sequence left, wrapping the first token to the end | A1 → M1 → M2 → A3 | `1.67e-8` |
| `P11_check_palindrome` | Repeat a flag indicating whether the sequence reads the same backwards | A1 → M1 → M2 → M3 → A4 → M4 → A5 → M5 → M6 | `0` |
| `P12_sorting` | Sort tokens in ascending order | M1 → A2 → M2 → A3 | `4.78e-8` |
| `A_prev_token_class` | Copy the previous token class; the first copies itself | A1 | `2.10e-8` |
| `B_histogram_then_repeated_class` | Count occurrences, then flag classes occurring more than once | A1 → M1 → M2 | `0` |

## Focused examples: A and B

Both examples below use the saved input `[1, 2, 1]`. Their variable timing is documented in the [Step 1 progression schedule](../tables/step1_progression_schedule.md).

### A: Previous token class

Input tokens represent categorical class IDs. Attention selects the preceding input position, with the first position selecting itself:

| Output position | Selected input position | Output class |
| --- | --- | --- |
| 0 | 0 | 1 |
| 1 | 0 | 1 |
| 2 | 1 | 2 |

The result is **`[1, 1, 2]`**. Its single computed variable appears after `A1`. The saved model and RASP answers agree on all five inputs.

### B: Histogram, then repeated-class flag

This program has two dependent computed variables:

```text
Input classes:                  [1, 2, 1]
histogram, after A1 → M1:        [2, 1, 2]
repeated_class, after M2:        [1, 0, 1]
```

The first stage counts occurrences of each class. The second reads those counts and returns `1` when a count exceeds one, otherwise `0`. These values explain the program's calculation; the JSON stores the intermediate state as residual arrays and output-channel scores.

The final result is **`[1, 0, 1]`**. The manual and model final scores are identical in all five saved cases.

## Inputs covered

Every program uses these same five inputs:

| Input | Situation covered |
| --- | --- |
| `[1, 2, 1]` | A repeated token and a palindrome |
| `[-2, 0, 3, 1]` | Negative, zero, and positive values in mixed order |
| `[5]` | A single-token sequence and boundary behavior |
| `[0, 0, 0, 0]` | All tokens equal |
| `[5, 4, 3, 2, 1, 0]` | Six tokens in descending order |

## How to read a JSON record

| Field | Meaning |
| --- | --- |
| `program`, `tokens` | Which program and input were run |
| `rasp_output` | Expected answer from the RASP evaluator |
| `model_decoded` | Decoded answer from the compiled Transformer |
| `manual_scores`, `model_scores` | Final output-channel scores from the manual pass and model |
| `max_abs_score_error` | Largest absolute difference between those final scores |
| `block_trace[].module` | Attention or MLP block that just ran |
| `block_trace[].residual` | Manual residual stream after that block |
| `block_trace[].output_scores` | Manual output-channel scores at that intermediate point |

## What the saved evidence establishes

The JSON directly supports agreement of the **final scores and decoded answers for all 70 sampled cases**. The saved notebook execution additionally reports **zero intermediate or final residual mismatches for every program**, with a largest residual difference of approximately `1.91e-6`.

The JSON contains manual intermediate residuals but does not include actual model intermediate residuals, residual comparison errors, or residual-lane labels. Those intermediate comparisons are therefore supported by the notebook's saved output. The results cover the five listed inputs per program; they are not an exhaustive check of every possible input.

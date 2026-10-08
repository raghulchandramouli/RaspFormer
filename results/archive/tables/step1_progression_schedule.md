# Focused Step 1 RASP progression

Tracr 1.0.0 (`9ce2b8c82b6ba10e62e86cf6f390e7536d4fd2cd`); 2/2 programs compiled and schedules extracted. Historical Step 1 snapshot. Step 2 tracing was subsequently completed; see [the trace notebook](../../../notebooks/02_trace.ipynb).

A treats each input token as a categorical class ID. At position zero it carries the current class; later positions receive the immediately previous class.

B computes each token class's histogram, then derives a binary repeated-class flag from that histogram.

## A_prev_token_class

1 transformer layer(s); 1 computed sequence variable(s).

| Variable | Residual lanes | First appears |
|---|---:|---|
| indices | 0–5 | input embedding |
| tokens | 15–22 | input embedding |
| prev_token_class_1 | 7–14 | after attention1 |

## B_histogram_then_repeated_class

2 transformer layer(s); 2 computed sequence variable(s).

| Variable | Residual lanes | First appears |
|---|---:|---|
| tokens | 17–24 | input embedding |
| histogram_4 | 0–6 | after mlp1 |
| repeated_class_3 | 15–16 | after mlp2 |


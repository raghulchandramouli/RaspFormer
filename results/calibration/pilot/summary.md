# Calibration pilot

Decision: **go**. 5 of 5 pilot pairs have at least two SIIT seeds per member at mean interchange accuracy ≥ 0.95; the rule needs 4. The pilot decides only whether the pairs train and the pipeline runs. Five pairs cannot support the L3 test, so the ratios below are descriptive.

## Training

| model_class   |   models |   token_accuracy |   lowest_accuracy |   mean_iia |
|:--------------|---------:|-----------------:|------------------:|-----------:|
| output-only   |       30 |            0.998 |             0.987 |      0.209 |
| siit          |       30 |            0.999 |             0.991 |      0.991 |

## Excluded models

| spec              | model_class   |   seed | reason                                          |
|:------------------|:--------------|-------:|:------------------------------------------------|
| pair-sort-b       | siit          |      1 | mean interchange accuracy 0.947 < 0.95          |
| pair-palindrome-a | output-only   |      2 | task accuracy not within 0.01 of the SIIT model |
| pair-palindrome-b | output-only   |      2 | task accuracy not within 0.01 of the SIIT model |

## Between-algorithm over between-seed distance, SIIT models

| pairing            |   cka |   effective_rank |   explained_variance_top1 |   explained_variance_top3 |   explained_variance_top5 |   participation_ratio |   procrustes |   spectral_entropy |   svcca |
|:-------------------|------:|-----------------:|--------------------------:|--------------------------:|--------------------------:|----------------------:|-------------:|-------------------:|--------:|
| count-greater-than | 1.52  |             2.49 |                      1.81 |                      2.76 |                      2.95 |                  2.01 |        1.27  |               2.64 |   0.79  |
| histogram          | 1.94  |             2.84 |                      3.09 |                      3.09 |                      2.71 |                  2.92 |        1.44  |               3.14 |   1.28  |
| palindrome         | 0.983 |             1.77 |                      1.28 |                      2.32 |                      2.51 |                  1.67 |        0.992 |               1.73 |   0.919 |
| reverse            | 1.81  |             3.19 |                      2.04 |                      1.95 |                      2.1  |                  2.4  |        1.56  |               3.22 |   1.46  |
| sort               | 2.19  |             3.94 |                      2.85 |                      3.58 |                      3.68 |                  3.08 |        1.65  |               3.93 |   1.52  |

## Summary over the five pairs

Output-loss-only twins of the two members share a seed, training data and target function, so twins with the same seed are the same model. Their ratio over all cross pairs is 2/3 by construction; `median_ratio_cross_seed` leaves same-seed cross pairs out and should be near one.

| model_class   | metric                  |   pairs |   median_ratio |   median_ratio_cross_seed |   identification_accuracy |   mean_between |
|:--------------|:------------------------|--------:|---------------:|--------------------------:|--------------------------:|---------------:|
| compiled      | participation_ratio     |       0 |        nan     |                    nan    |                   nan     |         3.5    |
| compiled      | spectral_entropy        |       0 |        nan     |                    nan    |                   nan     |         0.276  |
| compiled      | effective_rank          |       0 |        nan     |                    nan    |                   nan     |         5.07   |
| compiled      | explained_variance_top1 |       0 |        nan     |                    nan    |                   nan     |         0.0319 |
| compiled      | explained_variance_top3 |       0 |        nan     |                    nan    |                   nan     |         0.0513 |
| compiled      | explained_variance_top5 |       0 |        nan     |                    nan    |                   nan     |         0.0745 |
| compiled      | cka                     |       0 |        nan     |                    nan    |                   nan     |         0.206  |
| compiled      | procrustes              |       0 |        nan     |                    nan    |                   nan     |         0.556  |
| compiled      | svcca                   |       0 |        nan     |                    nan    |                   nan     |         0.0937 |
| siit          | participation_ratio     |       5 |          2.4   |                      2.38 |                     0.966 |         1.71   |
| siit          | spectral_entropy        |       5 |          3.14  |                      3.14 |                     0.966 |         0.379  |
| siit          | effective_rank          |       5 |          2.84  |                      2.84 |                     0.966 |         2.17   |
| siit          | explained_variance_top1 |       5 |          2.04  |                      2.01 |                     0.931 |         0.132  |
| siit          | explained_variance_top3 |       5 |          2.76  |                      2.77 |                     1     |         0.113  |
| siit          | explained_variance_top5 |       5 |          2.71  |                      2.71 |                     0.966 |         0.0771 |
| siit          | cka                     |       5 |          1.81  |                      1.81 |                     0.862 |         0.423  |
| siit          | procrustes              |       5 |          1.44  |                      1.46 |                     0.897 |         0.723  |
| siit          | svcca                   |       5 |          1.28  |                      1.3  |                     0.69  |         0.257  |
| output-only   | participation_ratio     |       5 |          0.667 |                      1    |                     0     |         0.476  |
| output-only   | spectral_entropy        |       5 |          0.667 |                      1    |                     0     |         0.0927 |
| output-only   | effective_rank          |       5 |          0.667 |                      1    |                     0     |         0.512  |
| output-only   | explained_variance_top1 |       5 |          0.667 |                      1    |                     0     |         0.0434 |
| output-only   | explained_variance_top3 |       5 |          0.667 |                      1    |                     0     |         0.0268 |
| output-only   | explained_variance_top5 |       5 |          0.667 |                      1    |                     0     |         0.0184 |
| output-only   | cka                     |       5 |          0.667 |                      1    |                     0     |         0.179  |
| output-only   | procrustes              |       5 |          0.667 |                      1    |                     0     |         0.34   |
| output-only   | svcca                   |       5 |          0.667 |                      1    |                     0     |         0.112  |

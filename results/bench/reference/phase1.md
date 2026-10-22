# Phase 1: InterpBench models under RaspFormer's detectors and stress suite

## Trained-model agreement with the compiled program

| Case | Model | L | H | d | Params | Held-out token acc | Eval token acc | Eval seq acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | compiled | 4 | 1 | 111 | 138336 | 1.0000 | 1.0000 | 1.0000 |
| 2 | trained | 4 | 4 | 56 | 156154 | 1.0000 | 1.0000 | 1.0000 |
| 7 | compiled | 1 | 1 | 26 | 2297 | 1.0000 | 1.0000 | 1.0000 |
| 7 | trained | 2 | 4 | 17 | 7535 | 1.0000 | 0.9996 | 0.9967 |
| 13 | compiled | 2 | 2 | 30 | 8268 | 1.0000 | 1.0000 | 1.0000 |
| 13 | trained | 2 | 4 | 20 | 10323 | 1.0000 | 1.0000 | 1.0000 |
| 21 | compiled | 4 | 1 | 66 | 80568 | 1.0000 | 1.0000 | 1.0000 |
| 21 | trained | 4 | 4 | 50 | 121079 | 1.0000 | 0.9996 | 0.9967 |
| 44 | compiled | 1 | 1 | 34 | 4377 | 1.0000 | 1.0000 | 1.0000 |
| 44 | trained | 2 | 4 | 24 | 15058 | 1.0000 | 1.0000 | 1.0000 |
| 124 | compiled | 3 | 1 | 48 | 15204 | 1.0000 | 1.0000 | 1.0000 |
| 124 | trained | 3 | 4 | 24 | 21986 | 1.0000 | 1.0000 | 1.0000 |

## Blind first-write detection F1

| Case | Model | Truth | change | pr | cka | all-positive | events/stages |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | compiled | schedule | 0.889 | 0.857 | 0.400 | 0.667 | 4/8 |
| 2 | trained | schedule | 0.667 | 0.667 | 0.667 | 0.667 | 4/8 |
| 2 | trained | circuit | 0.769 | 0.769 | 0.769 | 0.769 | 5/8 |
| 7 | compiled | schedule | 0.667 | 0.667 | 1.000 | 0.667 | 1/2 |
| 7 | trained | schedule | 0.400 | 0.400 | 0.500 | 0.400 | 1/4 |
| 7 | trained | circuit | 0.667 | 0.667 | 0.800 | 0.667 | 2/4 |
| 13 | compiled | schedule | 1.000 | 1.000 | 1.000 | 0.857 | 3/4 |
| 13 | trained | schedule | 0.857 | 0.857 | 1.000 | 0.857 | 3/4 |
| 13 | trained | circuit | 0.857 | 0.857 | 1.000 | 0.857 | 3/4 |
| 21 | compiled | schedule | 0.889 | 0.889 | 0.857 | 0.667 | 4/8 |
| 21 | trained | schedule | 0.667 | 0.727 | 0.667 | 0.667 | 4/8 |
| 21 | trained | circuit | 0.545 | 0.600 | 0.545 | 0.545 | 3/8 |
| 44 | compiled | schedule | 0.667 | 0.667 | 0.667 | 0.667 | 1/2 |
| 44 | trained | schedule | 0.400 | 0.000 | 0.000 | 0.400 | 1/4 |
| 44 | trained | circuit | 0.667 | 0.667 | 0.500 | 0.667 | 2/4 |
| 124 | compiled | schedule | 0.750 | 0.750 | 0.000 | 0.667 | 3/6 |
| 124 | trained | schedule | 0.667 | 0.667 | 0.667 | 0.667 | 3/6 |
| 124 | trained | circuit | 0.909 | 0.909 | 0.909 | 0.909 | 5/6 |

Unweighted means over cases:

| Model | Truth | Detector | mean F1 | mean all-positive F1 |
| --- | --- | --- | --- | --- |
| compiled | schedule | change | 0.810 | 0.698 |
| compiled | schedule | cka | 0.654 | 0.698 |
| compiled | schedule | pr | 0.805 | 0.698 |
| trained | circuit | change | 0.736 | 0.736 |
| trained | circuit | cka | 0.754 | 0.736 |
| trained | circuit | pr | 0.745 | 0.736 |
| trained | schedule | change | 0.610 | 0.610 |
| trained | schedule | cka | 0.583 | 0.610 |
| trained | schedule | pr | 0.553 | 0.610 |

## Stress suite: schedule F1 beside task accuracy (trained models, schedule truth)

| Case | Transform | Fraction | Token acc | change F1 | pr F1 | cka F1 |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | rotation | 1.0 | 1.000 | 0.667 | 0.667 | 0.667 |
| 2 | compression-pca | 0.25 | 0.154 | 0.667 | 0.667 | 0.545 |
| 2 | compression-pca | 0.5 | 0.692 | 0.667 | 0.667 | 0.667 |
| 2 | compression-pca | 0.75 | 0.959 | 0.667 | 0.667 | 0.667 |
| 2 | compression-pca | 1.0 | 1.000 | 0.667 | 0.667 | 0.667 |
| 2 | compression-random | 0.5 | 0.162 | 0.667 | 0.667 | 0.667 |
| 7 | rotation | 1.0 | 1.000 | 0.400 | 0.400 | 0.500 |
| 7 | compression-pca | 0.25 | 0.293 | 0.400 | 0.500 | 0.500 |
| 7 | compression-pca | 0.5 | 0.362 | 0.400 | 0.000 | 0.667 |
| 7 | compression-pca | 0.75 | 0.365 | 0.400 | 0.500 | 0.500 |
| 7 | compression-pca | 1.0 | 1.000 | 0.400 | 0.400 | 0.500 |
| 7 | compression-random | 0.5 | 0.087 | 0.400 | 0.500 | 0.667 |
| 13 | rotation | 1.0 | 1.000 | 0.857 | 0.857 | 1.000 |
| 13 | compression-pca | 0.25 | 0.642 | 0.857 | 0.667 | 0.857 |
| 13 | compression-pca | 0.5 | 0.706 | 0.857 | 0.857 | 1.000 |
| 13 | compression-pca | 0.75 | 0.763 | 0.857 | 0.857 | 1.000 |
| 13 | compression-pca | 1.0 | 1.000 | 0.857 | 0.857 | 1.000 |
| 13 | compression-random | 0.5 | 0.501 | 0.857 | 0.857 | 1.000 |
| 21 | rotation | 1.0 | 1.000 | 0.667 | 0.727 | 0.667 |
| 21 | compression-pca | 0.25 | 0.816 | 0.667 | 0.667 | 0.600 |
| 21 | compression-pca | 0.5 | 0.862 | 0.667 | 0.667 | 0.667 |
| 21 | compression-pca | 0.75 | 0.953 | 0.667 | 0.667 | 0.667 |
| 21 | compression-pca | 1.0 | 1.000 | 0.667 | 0.727 | 0.667 |
| 21 | compression-random | 0.5 | 0.619 | 0.667 | 0.667 | 0.667 |
| 44 | rotation | 1.0 | 1.000 | 0.400 | 0.000 | 0.000 |
| 44 | compression-pca | 0.25 | 0.076 | 0.400 | 0.400 | 0.400 |
| 44 | compression-pca | 0.5 | 0.069 | 0.400 | 0.400 | 0.400 |
| 44 | compression-pca | 0.75 | 0.119 | 0.400 | 0.000 | 0.000 |
| 44 | compression-pca | 1.0 | 1.000 | 0.400 | 0.000 | 0.000 |
| 44 | compression-random | 0.5 | 0.105 | 0.400 | 0.400 | 0.000 |
| 124 | rotation | 1.0 | 1.000 | 0.667 | 0.667 | 0.667 |
| 124 | compression-pca | 0.25 | 0.031 | 0.667 | 0.857 | 0.667 |
| 124 | compression-pca | 0.5 | 0.927 | 0.667 | 0.750 | 0.667 |
| 124 | compression-pca | 0.75 | 1.000 | 0.667 | 0.667 | 0.667 |
| 124 | compression-pca | 1.0 | 1.000 | 0.667 | 0.667 | 0.667 |
| 124 | compression-random | 0.5 | 0.422 | 0.667 | 0.667 | 0.667 |

Unweighted means over cases (trained models, schedule truth):

| Transform | Fraction | Detector | mean token acc | mean F1 |
| --- | --- | --- | --- | --- |
| compression-pca | 0.25 | change | 0.335 | 0.610 |
| compression-pca | 0.25 | cka | 0.335 | 0.595 |
| compression-pca | 0.25 | pr | 0.335 | 0.626 |
| compression-pca | 0.5 | change | 0.603 | 0.610 |
| compression-pca | 0.5 | cka | 0.603 | 0.678 |
| compression-pca | 0.5 | pr | 0.603 | 0.557 |
| compression-pca | 0.75 | change | 0.693 | 0.610 |
| compression-pca | 0.75 | cka | 0.693 | 0.583 |
| compression-pca | 0.75 | pr | 0.693 | 0.560 |
| compression-pca | 1.0 | change | 1.000 | 0.610 |
| compression-pca | 1.0 | cka | 1.000 | 0.583 |
| compression-pca | 1.0 | pr | 1.000 | 0.553 |
| compression-random | 0.5 | change | 0.316 | 0.610 |
| compression-random | 0.5 | cka | 0.316 | 0.611 |
| compression-random | 0.5 | pr | 0.316 | 0.626 |
| rotation | 1.0 | change | 1.000 | 0.610 |
| rotation | 1.0 | cka | 1.000 | 0.583 |
| rotation | 1.0 | pr | 1.000 | 0.553 |

# Independent review

Final audit by `gpt-6-astra` passed. Earlier model and scientific reviews used `gpt-6-astra` and `gpt-6-sol`.

The reviewers checked imported execution, padding and causal masks, basis transformations, actual residual compression, sequence separation, prediction freezing, checkpoint reproduction, report claims, and evidence paths.

The final audit independently recomputed all twelve patch-stage choices from calibration activations. It checked 24 frozen prediction hashes, 288 transformed detector rows, manifest-summary agreement, and 36 checkpoint hashes. The original model reviewer replayed all 36 saved models against their exact predictions and independent RASP targets.

Accepted corrections added an always-positive write baseline, disclosed shared calibration PCA in decoder validation, moved patch-stage selection entirely to calibration activations, and labeled the transformed recovery table's held-out program population. Two maintenance suggestions strengthened the verifier's argmax and checkpoint-coverage checks. The verifier passes after those changes.

The evidence remains limited to the recorded tasks and distributions. Write-transition recovery does not recover complete algorithms. Learned models have no known internal causal circuit. Compression can change behavior. No general variable-lifetime recovery is claimed.

See [the generated report](research-validation.md), [verification evidence](../validation/reference/verification.json), and [decision trail](decisions.tsv).

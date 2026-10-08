# RaspFormer: PNG figure guide

13 figures from the illustrated Steps 1-5 report. All PNGs are 300 dpi. Eight figures were added in this expansion. The images redraw saved results; no experiments were rerun.

Files are numbered in PDF reading order. Each section explains the axes or encodings and the main interpretation. For complete procedures, formulas and scope, read RaspFormer_Steps_1_to_5.pdf.

## 01 · Compiler profile

![Compiler profile](assets/compiler_profile.png)

Figure 1. Compiler properties are different notions of complexity. P11 is deepest; P12 is widest. Program numbers are identifiers, not an ordered difficulty scale. Source: R1.

Read across each program row to compare compiled layers, computed variables and residual lanes. Each panel has its own scale. These properties describe different aspects of the compiled model; the program IDs are not a difficulty ranking.

## 02 · First writes for palindrome checking and sorting

![First writes for palindrome checking and sorting](assets/extra_early_write_schedule.png)

Figure 2. Compiler allocations for P11 and P12, with numeric expression suffixes removed for readability. Source: R1/manifest.json, variable and program records.

**How to read it.** Rows are computed variables. Columns are attention (A) and MLP (M) stages. Each dot marks the variable's first scheduled write; orange means attention and teal means MLP.

**What it shows.** P11 builds length, opposite positions, reversed tokens, mirror matches and a mismatch count before writing the palindrome answer at Layer 6 MLP. P12 writes three variables, ending with sorted values at Layer 3 attention. P12 is wider, but P11 has more layers and variables.

**The boundary.** Blank cells do not mean the model is inactive. The chart records allocations, not measured activation onset, variable lifetimes or last uses.

## 03 · Actual residual lane values for the B trace

![Actual residual lane values for the B trace](assets/extra_early_residual_trace.png)

Figure 3. Selected lanes from B's saved manual trace. Sources: R2/traces.json and the residual labels in R2/manifest.json.

**How to read it.** A row is an input position. Count 2 is a value of 1 in the column labelled 2. The three panels follow the active blocks in order; BOS and unrelated lanes are omitted.

**What it shows.** Helper values appear first. The count lanes then encode [2, 1, 2]; the class lanes later encode [1, 0, 1]. Earlier counts persist beside the new labels. Fractions are rounded displays of saved values.

**Why it matters.** The helper has activity before a named count is written. Geometry can reflect compiler helper state as well as semantic variables. This picture documents one input's trace.

## 04 · Geometry in example B

![Geometry in example B](assets/geometry_b.png)

Figure 4. Saved computed-view geometry for B. Gray CKA cells are undefined. Nonzero dimensions and high similarity describe variance structure; they do not by themselves identify which variable was computed. Source: R3.

The left panel tracks computed-view PR across stages. On the right, each heatmap cell compares the same target observations at two stages; darker blue means higher CKA. Gray is undefined, not zero. A change in geometry alone does not identify a computation.

## 05 · Balanced target counts and B position allocation

![Balanced target counts and B position allocation](assets/extra_geometry_sampling.png)

Figure 5. Saved Step 3 observation counts and the position allocation for B. Teal bars are the main program set; orange bars are the focused examples. Source: R3/balance_audit.csv and probes_B_histogram_then_repeated_class.csv.

**How to read it.** Each top bar counts selected target positions, including repeated selections. The bottom grid splits B's targets by output class and position. Every row sums to 64; cells contain 10 or 11 observations because 64 does not divide evenly by six.

**What it shows.** P07 and P08 have 15 reachable output classes, so each gets 960 targets. B has two classes and gets 128. The balance rule is consistent even though the program totals differ.

**Why it matters.** Balanced outputs do not imply identical inputs across programs. Step 5 addresses this by giving every main program the same 20,298 candidate-position pairs. Neither design makes observations from the same input sequence independent.

## 06 · PCA spectra for three single-variable programs

![PCA spectra for three single-variable programs](assets/extra_geometry_spectrum.png)

Figure 6. Final computed-view PCA spectra under the Step 3 balanced probes. Each panel is one program with one computed RASP variable. Sources: R3/pca_spectra.npz and layer_metrics.csv.

**How to read it.** A bar is a principal component, and its height is its percentage of total variance. The bars in each panel sum to 100%. A tall single bar means variance is concentrated in one direction; many equal bars mean it is spread evenly.

**What it shows.** P01's six categorical output values occupy six one-hot lanes. Centering removes one direction, leaving five equal directions of 20% each and PR = 5. P02 has one direction with all the variance, so PR = 1. P03 has twelve equal directions of about 8.33% each, so PR = 12.

**The interpretation.** All three programs compute one variable. Their different PRs come from their encoded value distributions. This is why Step 4 compares observed PR with the PR of the encoded RASP values on the same probes.

For k equally weighted one-hot categories, centering leaves k - 1 equal nonzero directions. Under that specific distribution, PR = k - 1.

Different category frequencies or a different encoding can change the spectrum. The numbers here describe these saved probes and representations.

## 07 · B intervention with original control and donor outputs

![B intervention with original control and donor outputs](assets/extra_alignment_causal_b.png)

Figure 7. The saved B intervention at probe 0, position 4, using donor probe 23. Source: R4/interventions.csv.

**How to read it.** Follow the highlighted fifth position down the page. The donor supplies a count of 1, not its input token. The patch occurs after the count is written; the downstream classifier then reads the replacement.

**What it shows.** The control preserves all six labels. The donor changes only the intended label from 1 to 0, matching RASP's substituted computation with zero output-score error.

**Scope.** This is one of 128 B donor checks, each paired with a control. Donors were selected to change the answer. The result checks this dependency in the verified manual circuit; it does not estimate general variable importance.

## 08 · Observed and known-value participation ratios

![Observed and known-value participation ratios](assets/alignment_pr.png)

Figure 8. P01 has one computed variable but final PR = 5 under these balanced categorical probes. P12 shows larger benchmark gaps. The plots compare encoded distributions, not PR with a variable count. Source: R4.

The horizontal axis follows stage order; the vertical axis is participation ratio. Teal is the observed residual and orange is the encoded-RASP benchmark. Overlapping curves mean PR agreement on these probes, not identical vectors. P01 reaches PR 5 while computing only one variable.

## 09 · P12 output lane correlation and magnitude agreement

![P12 output lane correlation and magnitude agreement](assets/extra_alignment_lane_tolerance.png)

Figure 9. P12's eight sorting-output lanes at their first write, L3/Attn, under balanced probes. Source: R4/lanes.csv.

**How to read it.** Each row represents one categorical output lane, not a token position. The left panel counts observations within tolerance. The right panel shows each lane's worst absolute discrepancy; its numbers are multiplied by 10^-3. The dashed line is the largest allowed tolerance for a one-hot value.

**What it shows.** Correlation exceeds 0.999999999996 in every lane, yet each passes the value check on only 87.5% of its 512 observations. The maximum error, 0.000852227, is over four times the largest permitted tolerance of 0.0002.

**The distinction.** Correlation measures co-variation and can remain almost perfect despite an amplitude discrepancy. All decoded answers still pass. The same eight rows recur at L3/MLP, accounting for the 16 lane/stage discrepancies reported above.

## 10 · Sampling comparison

![Sampling comparison](assets/sampling_comparison.png)

Figure 10. Matched and balanced agreement summaries, computed view. Missing CKA points are undefined. PR/CKA axes use a symmetric log scale with a linear region near zero; the lane axis is zoomed. Source: R5.

Compare the teal matched-input dot and orange balanced-input cross within each program row. Lower PR/CKA errors and higher lane agreement mean closer benchmark agreement. PR/CKA use a symmetric logarithmic scale with a linear region near zero; lane agreement uses a zoomed scale. Missing CKA points are undefined, not zero error. Designs differ in input weighting and sample count.

## 11 · CKA availability under shared and class-balanced inputs

![CKA availability under shared and class-balanced inputs](assets/extra_comparison_cka_coverage.png)

Figure 11. Availability of adjacent observed-versus-reference CKA pairs in the computed view. Source: R5/comparison.csv.

**How to read it.** Each bar contains all transitions for one program. Teal pairs enter the CKA error mean; orange and gray transitions do not. The right-hand fraction supplies the actual count, because programs have different numbers of transitions.

**What it shows.** Both designs have 24 usable pairs out of 54. Shared inputs produce 14 availability mismatches, compared with two under class balance. P07, P08, P10 and P11 each contribute three additional mismatches. With the full residual view, all 54 pairs are defined in both designs.

**Why the missing values matter.** A tiny CKA error describes only the teal portion. Undefined is not zero similarity. The implementation accepts any positive covariance norm; the saved summaries alone cannot establish whether very small nonzero variation is meaningful signal or numerical variation.

## 12 · PR agreement in full and computed residual views

![PR agreement in full and computed residual views](assets/extra_comparison_residual_views.png)

Figure 12. Matched-input PR error under two coordinate selections. Every program uses the same 20,298 observations. Source: R5/comparison.csv.

**How to read it.** Navy circles include the full residual; teal diamonds remove the named input lanes. Lower error means closer PR agreement with the encoded RASP benchmark. Connecting lines join two views of one program, not a trajectory or confidence interval.

**What it shows.** P12 changes from 0.4255 in the full view to 2.1339 in the computed view. P07, P08 and P10 have full-view errors below 5 × 10^-7, while their computed-view errors are about 0.5. P09 moves slightly in the opposite direction.

**Interpretation.** Selecting coordinates changes how variance is distributed, so it can change PR agreement. The computed view still contains helper lanes. These numbers do not identify the magnitude or cause of early variation, and they do not imply incorrect decoded outputs.

## 13 · Complexity comparison

![Complexity comparison](assets/complexity_pr.png)

Figure 13. Matched computed-view PR error against measured complexity. Computed-variable count and dependency depth are identical for every program in this set, so they share one axis here. The notebook retains all three declared axes. Source: R5.

Each labelled point is a main program under matched inputs and the computed view. The vertical axis is mean absolute PR error; the horizontal axes are compiler complexity measures. Variable count equals dependency depth in this set. This is a descriptive comparison of twelve programs, not evidence that complexity causes error.

## Source register

All paths below are relative to the repository's `results/` directory. The dated runs are the numerical evidence; reference folders and the archive provide historical context.

| Key | Evidence folder | Main files |
| --- | --- | --- |
| R1 | `programs/20261007T071415_1d4b7299/` | `manifest.json`, `schedule.md` |
| R2 | `trace/20261007T071435_2694c39c/` | `manifest.json`, `traces.json` |
| R3 | `geometry/20261007T071505_930a7a88/` | `layer_metrics.csv`, probe and CKA tables, `pca_spectra.npz` |
| R4 | `alignment/20261008T061604_c489bd97/` | `stages.csv`, `lanes.csv`, `variables.csv`, `values.csv`, `interventions.csv` |
| R5 | `comparison/20261008T064602_ac76cb03/` | `comparison.csv`, `matched/`, `balanced/`, sampling and CKA-write summaries |

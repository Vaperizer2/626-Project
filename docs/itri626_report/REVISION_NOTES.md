# Three-part adversarial presentation review (2026-11-05)

Reviewed the report at `6db0fb5`, then applied focused presentation changes. The instruction was to preserve
scientific content and make the report read and print like a well prepared article. This is three review passes,
not a claim of three independent reviewers: a Codex visual reviewer inspected all original pages, and the parent
performed the prose and reference passes. The configured Claude prose reviewer failed immediately with an API rate
limit; no findings were attributed to it and no replacement reviewer was launched.

## 1. Prose and article structure

Verified against `git show 6db0fb5:docs/itri626_report/main.tex` and the rendered PDF:

| Finding | Applied response |
|---|---|
| The abstract's “each with a trained linear head” could attach to the entire A–D list. | Named C/D explicitly as the two frozen encoders with fitted heads. |
| The abstract described score gaps as effects of pretraining although the discussion acknowledges confounding. | Expressed the same numerical gaps as comparisons between B, C and D. |
| “Leakage-free” implied a stronger guarantee than the residual-duplicate limitation. | Used “duplicate-aware” in the subtitle/scope and “detected” near-duplicate clusters in the abstract. |
| Dense prose and awkward date wrapping distracted from the article presentation. | Kept the section structure and content, tightened a few phrases and placed the title-page date on its own line. |
| Appendix floats interrupted “Measured accuracy…” across a whole intervening page. | Flushed floats before the detailed limitations; began the following appendix on a fresh page. |

No model selection, reported outcome, image selection or overall conclusion was changed. Recommendations to
change the scientific scope or fit a new publication template were dropped: the user's request was a polish pass.

## 2. Figures and tables

The independent reviewer rendered all 27 original pages using `pdftoppm -r 110` and measured fonts with
`pdftohtml -xml -hidden -i -zoom 1`. Confirmed findings and final dispositions:

| Original evidence | Applied response |
|---|---|
| Physical p11: Figure 6 count labels approximately 6 pt, 20×20 matrices about 51 mm square. | Moved full counts to Figures A.1/A.2, two larger matrices per page; 8.5 pt counts, numbered axes and a readable class key, with one shared count scale. |
| Physical pp14,18–19: repeated model P/R/F1 and confusion groups lacked body guides. | Added 0.3 pt subdued vertical separators between model groups in Table 5 and Tables A.1–A.3. No outer boxes or per-cell grid. |
| Physical p20: Table B.1 used 7–8 pt text with four long model columns. | Split into two continued parts with repeated model headers, 9 pt text, more row spacing and group rules. Every setting remains. |
| Physical pp4,13–14,22: example, gallery and disagreement labels were marginal. | Enlarged the class-example figure; reset original gallery titles/confidences at 8.5 pt and reused exactly the same embedded photos; gave disagreements the full line width. |
| Crowded body figures competed for space. | Retained C's examples in the body, moved A/B's full examples to Appendix A, and kept all four matrices and agreement counts as directly linked supporting figures. |

The original Excalidraw scenes/exports and all notebook-result exports remain unchanged. New report-local PDFs
and their reproduction script are in `figures/` and `make_presentation_figures.py`. The class numbering is asserted
against `split_info.json`. No raw images or model execution are used for this artwork refresh.

## 3. References and formatting consistency

Checked all 33 rendered bibliography entries against Crossref, publisher/proceedings pages, author arXiv records
or official project sources. Resolved initial Crossref 429 responses through later individual requests. OpenReview
forum pages presented a browser challenge; author records confirmed the relevant ICLR publication information.
The Kaggle URL responds but its live card exposed no readable metadata, so its recorded citation/access date was
preserved rather than re-inventing metadata. This is the remaining source-access limitation.

Added visible DOI/source links to the IEEE-style bibliography. Completed FastViT's missing pages and DOI and
normalised its venue abbreviation. Its DOI-registered IEEE pagination is 5762–5772; CVF's accepted-version
pagination differs. Likewise ConvNeXt's existing 11966–11976 is the DOI-registered version and was correctly
retained despite the different CVF accepted-version page range. Other identities and numerical metadata matched.

| Citation key | Check and primary source |
|---|---|
| `shao2025electronic` | [Crossref: title/authors, publication year and pages/article number matched](https://doi.org/10.1109/uv63228.2024.11189118) |
| `zhou2025electronic` | [Crossref: title/authors, publication year and pages/article number matched](https://doi.org/10.1142/9789819801473_0011) |
| `ling2023pcb` | [Crossref: title/authors, publication year and pages/article number matched](https://doi.org/10.1109/access.2023.3245093) |
| `kaggle_dataset` | [URL responds; live card unreadable; retained recorded citation](https://www.kaggle.com/datasets/mdfaisalahmed025/electronic-component-recognizer) |
| `radford2021clip` | [PMLR: authors, title, 2021, vol. 139, 8748–8763](https://proceedings.mlr.press/v139/radford21a.html) |
| `northcutt2021confident` | [Crossref: title/authors, publication year and pages/article number matched](https://doi.org/10.1613/jair.1.12125) |
| `pedregosa2011sklearn` | [JMLR: authors, title, 2011, vol. 12, 2825–2830](https://jmlr.org/papers/v12/pedregosa11a.html) |
| `barz2020cifair` | [Crossref: title/authors, publication year and pages/article number matched](https://doi.org/10.3390/jimaging6060041) |
| `kapoor2023leakage` | [Crossref: title/authors, publication year and pages/article number matched](https://doi.org/10.1016/j.patter.2023.100804) |
| `wightman2019timm` | [Official project identity and URL](https://github.com/huggingface/pytorch-image-models) |
| `simonyan2014vgg` | [Author paper: title/authors; ICLR 2015 publication retained](https://arxiv.org/abs/1409.1556) |
| `ioffe2015bn` | [PMLR: authors, title, 2015, vol. 37, 448–456](https://proceedings.mlr.press/v37/ioffe15.html) |
| `he2015delving` | [Crossref: title/authors, publication year and pages/article number matched](https://doi.org/10.1109/ICCV.2015.123) |
| `liu2022convnext` | [Crossref: title/authors, publication year and pages/article number matched](https://doi.org/10.1109/cvpr52688.2022.01167) |
| `he2016resnet` | [Crossref: title/authors, publication year and pages/article number matched](https://doi.org/10.1109/cvpr.2016.90) |
| `russakovsky2015imagenet` | [Crossref: title/authors, publication year and pages/article number matched](https://doi.org/10.1007/s11263-015-0816-y) |
| `kumar2022finetuning` | [Author record: authors/title and ICLR 2022 oral](https://arxiv.org/abs/2202.10054) |
| `yosinski2014transferable` | [NeurIPS proceedings identity/year](https://proceedings.neurips.cc/paper/2014/hash/532a2f85b6977104bc93f8580abbb330-Abstract.html) |
| `kornblith2019transfer` | [Crossref: title/authors, publication year and pages/article number matched](https://doi.org/10.1109/cvpr.2019.00277) |
| `dosovitskiy2020vit` | [Author record: authors/title and ICLR camera-ready version](https://arxiv.org/abs/2010.11929) |
| `zhai2023siglip` | [Crossref: authors/title, ICCV 2023, 11941–11952](https://doi.org/10.1109/ICCV51070.2023.01100) |
| `faghri2025mobileclip2` | [Author/publisher records: authors/title, TMLR 2025](https://machinelearning.apple.com/research/mobileclip2) |
| `vasu2023fastvit` | [Crossref: authors/title, ICCV 2023, 5762–5772](https://doi.org/10.1109/ICCV51070.2023.00532) |
| `efron1979bootstrap` | [Crossref: title/authors, publication year and pages/article number matched](https://doi.org/10.1214/aos/1176344552) |
| `mcnemar1947` | [Crossref: title/authors, publication year and pages/article number matched](https://doi.org/10.1007/BF02295996) |
| `cubuk2020randaugment` | [Crossref: authors/title, CVPRW 2020, 3008–3017](https://doi.org/10.1109/CVPRW50498.2020.00359) |
| `loshchilov2019adamw` | [Author record: authors/title, ICLR 2019](https://arxiv.org/abs/1711.05101) |
| `szegedy2016inception` | [Crossref: title/authors, publication year and pages/article number matched](https://doi.org/10.1109/cvpr.2016.308) |
| `loshchilov2017sgdr` | [Author record: authors/title, ICLR 2017](https://arxiv.org/abs/1608.03983) |
| `paszke2019pytorch` | [NeurIPS proceedings identity/year](https://proceedings.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html) |
| `astral2024uv` | [Official project identity and URL](https://docs.astral.sh/uv/) |
| `chow1970reject` | [Crossref: title/authors, publication year and pages/article number matched](https://doi.org/10.1109/tit.1970.1054406) |
| `geifman2017selective` | [NeurIPS proceedings: authors/title, vol. 30, 2017](https://proceedings.neurips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html) |

Formatting references inspected: [Springer IJCV author guidance](https://link.springer.com/journal/11263/submission-guidelines),
[Zhai et al., SigLIP, author PDF](https://arxiv.org/pdf/2303.15343) and
[Liu et al., ConvNeXt, author PDF](https://arxiv.org/pdf/2201.03545). The examples informed label sizes, grouping
and caption placement. The existing Charter single-column course layout and IEEE bibliography were retained;
this is not a claim of compliance with a specific Springer journal template.

## Final verification

- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`: passes; no LaTeX/BibTeX warnings, undefined
  references, overfull or underfull boxes.
- All 32 final PDF pages rendered with Poppler at 120 dpi and visually inspected; larger views checked the
  reworked matrices, gallery labels, table separators, continued settings and appendix transitions.
- All 499 `generated/key_numbers.json` values and provenance entries are identical to `6db0fb5`.
- Pixel hashes of all 12 photos in each A/B/C/D gallery match the original PDFs (48 photos total).
- `pdffonts`: all fonts embedded. `git diff --check`: passes.
- No changes under `results/`, `exports/`, `apps/`, `crates/` or packaging. No training, holdout inference,
  main-checkout edits or release work.

The PDF now has 32 physical pages. The conclusion ends on printed page 14 (physical page 15), preserving the
15-page body budget. References begin later on that page and continue through printed page 16. Appendices use
printed pages 17–31; the extra supporting pages give the original evidence readable space.

---

# Four-model report polish (2026-11-05)

The current source is `main.tex`, integrated from the three-model course report on main `17aff8b` and the
historical proposals in `../model_d_inserts/`. The wider study in `../report/REPORT.md` is a separate document.

- Integrated D (frozen MobileCLIP2-S3 + linear head) across methods, preprocessing, fairness, metrics, learning
  curves, per-class F1, confusion matrices, appendix examples and full per-class results.
- Extended `make_tables.py` to read committed D/R3 results and all six paired comparisons. The existing number
  provenance entries remain identical; no recorded experiment metrics or prediction files changed.
- Corrected the proposal's R3 identification: `R3_small_fused` is the packaged Small classifier, refitted on all
  development rows and fused with zero-shot scores. It is not the course D head or a same-training-split reference.
- Replaced the workflow with an actual Excalidraw MCP scene; added an evaluation-protocol diagram. Editable scenes,
  MCP inputs and official-library SVG/PDF exports are in `diagrams/`. No Mermaid substitution was used.
- Distinguished C/D's overlapping individual intervals from their paired result; labelled unadjusted McNemar
  p-values. Removed claims that log-loss proves calibration or that practical recipe comparisons isolate causality.
- Qualified CPU/GPU cost claims by hardware, precision, runtime, batch and excluded preprocessing; added the
  same-machine C-tower anchor. D and Small differ; Fast's INT8 engine is not the 308 px fp16 benchmark.
- Moved the wide hyperparameter table and detailed supporting/limitation material to the appendices, retained a
  concise limitations section, updated references and captions, and removed stale training commands from the routine
  report build. Both holdouts remain spent.
- Marked older inserts as historical, refreshed the rubric map and corrected the efficiency study's stale opening
  claim that no test scoring had occurred after development selection.

Validation: generated tables rebuilt from committed predictions/metrics; existing macro entries compared with HEAD
and unchanged; all figure/table inputs present; editable scenes parse as standard Excalidraw documents. `latexmk`
completed with no warnings, undefined references, overfull or underfull boxes. The PDF was rendered with Poppler and
all pages visually inspected, with larger views of the diagrams, metrics, matrices and appendix tables.

The final PDF has 27 pages: title plus sections 1–7 occupy 15 physical pages (printed pages 1–14 after the title);
references occupy printed pages 15–16 and supporting appendices 17–26. This follows the earlier report's body-budget
convention; references and appendices are additional. Lecturer approval for D as an extra family and the group's
slides/submission arrangements remain outside this edit. Main checkout, model assets and releases were untouched.

---

# ITRI626 report: revision notes (2026-10-20)

This revision applies two reviews of `main.pdf` at commit b5c4fea: the visual/layout review (**V**) and the
critical/content review (**C**). No result, number or claim changed in substance, and no experiment was re-run.
Every new sentence uses the existing macros in `generated/numbers.tex`. New facts were checked against repository
files: the 7-fold min-TVD test selection in `tools/build_dataset_v2.py` l.507–520, the crop fractions 0.875/0.9 in
`results/itri626/runs/*.json`, the Kaggle access date from `data/v2/manifest.json` `created_utc` = 2026-10-17, and
the "test never used by the benchmark" claim from `scripts/analysis_tables.py` (`dev` rows only).
`generated/key_numbers.json` differs from the previous build in formatting only: `vramB` 2312 → 2,312 and
`fpMBC` 1633.6 → 1,633.6.

**Result:** 23 PDF pages. The body is the title page plus sections 1–7 and ends at the foot of PDF page 15
(printed page 14). References are on printed pp. 15–16 and the appendices on pp. 17–22. The build has 0 LaTeX
warnings, 0 BibTeX warnings, 0 overfull or underfull boxes and 0 undefined references, and the duplicate
`page.1` destination is gone.

Where the reviews overlapped, the fix was applied once (cross-referenced as "= V…/C…"). The table also records
which option was chosen where they conflicted.

## Visual / layout review (V)

| Item | Status | What was done / why not |
|---|---|---|
| V P1-1 figure text 2.6–4.5 pt | applied | Every report figure is drawn at its printed width (`TEXTW = 6.30` in) and included at that width (Fig 1 0.64 tw, Fig 5 0.8 tw, Fig 8 0.36 tw, Fig 9 0.61 tw, all others 1.0 tw). The font ladder is (8.5, 7.5, 7) and no explicit size is below 6.5 pt except the confusion-matrix cells (6 pt). The checks of `git diff results/itri626/*.json *.csv preds` are empty. |
| V P1-2 galleries unreadable | applied | `fig_gallery`: 6.3 in wide, two- or three-line titles at 6.5 pt (`true → pred`), a 1.5 pt red frame on errors, no suptitle. Fig 7 is its own `[p]` float. Figs 8 and 9 are a separate float in §5.5, next to their first citation. |
| V P1-3 confusion matrices | applied (= C P2-18) | New `fig_confusion_grid` gives a 2×2 grid with one shared count scale for A/B/C, a separate 0–100 % scale for C row-normalised and no `trim`. Conflict: C suggested separate counts-only PDFs; the grid was chosen because it also fixes the non-comparable scales. The per-model `fig06_confusion_*` files are kept and now use the same shared `vmax`. Figures 5 and 6 share one float page. |
| V P1-4 appendix floats drift | applied | `\FloatBarrier` before each appendix section; the Appendix B sentence is reordered to follow float order. |
| V P1-5 workflow diagram | applied (= C P2-6) | The tested three-row TikZ from the review, with model colours. The box height was trimmed slightly (15.5/12/7 mm) to stay within the page budget. Model A is labelled "VGG-style CNN" and the flag "CLIP off-topic flag" (C P2-7). |
| V P1-6 duplicate `page.1` | applied | `\pagenumbering{Alph}` / `{arabic}` around the title page. |
| V P1-7 en dash in `--check`, `--steps` | applied | `\file{--check}` and `\texttt{-{}-steps …}`; pdftotext now gives "--". |
| V P1-8 title page | applied (= C P3-1) | Two-level title, rules, a group-member table, a lecturer line and the abstract at the foot. The `[University name] · [Faculty / School]` placeholders are left for the user. |
| V P2-1 XCharter + newtxmath + zi4 | applied | The `vvarbl` option was dropped because the installed newtxmath rejects it. |
| V P2-2 parskip | applied, modified | `parskip` package with `skip=4pt plus 1pt` (C P3-10 suggested 3 pt plus 1 pt). The review's 0.5 baselineskip pushed the body past 15 pages. |
| V P2-3 doubled space before units, % | applied (= C P1-7) | `\xspaceaddexceptions{\,\%}`. |
| V P2-4 paths break mid-word | applied (= C P2-11) | Conflict: V kept `xurl` with restricted breaks; C proposed `url`. Chose `url` with `\UrlBreaks` at `/ _ - .`, `\Urlmuskip`, and a mono font at 92 % of the surrounding size. |
| V P2-5 visible links + PDF metadata | applied (= C P3-11) | `colorlinks` in navy `accent` (#1F3A5F), the same colour as the headings; title, author, subject and bookmarks set. |
| V P2-6 emergencystretch, widow/club | applied | Both, plus `\textfloatsep` tightened slightly. |
| V P2-7 table font sizes | applied | Body tables use `\footnotesize`. Exception: Table 3 (hyperparameters) is `\scriptsize` for the page budget; it is also regrouped (P2-19). Table 6 fits at `\footnotesize` with "MCU board". Duplicate size commands were removed. |
| V P2-8 Table 1 dense | applied | `tabularx`, two-level Cleaning/Split header, `\addlinespace` every 5 rows, short caption, column legend as a `\scriptsize` note under the table, "Flag." column. |
| V P2-9 floats before their anchor | applied | `flafter`; Fig 2 is `[tb]`. |
| V P2-10 "Section A" → "Appendix A" | applied (= C P2-4.1) | `\crefalias{section}{appendix}`. |
| V P2-11 orange = model A = wrong | applied (= C P2-15) | Errors use a dark red frame (#B2182B) and ✗ marks; correct predictions get a green ✓. Model colours are defined in LaTeX, and a coloured square (`\msq`) appears before the model names in Tables 3, 4, 6, A.1, B.2 and B.3. |
| V P2-12 learning curves | applied (= C P2-14) | Conflict on layout: V kept 3×2, C proposed 2×3. Chose C's 2 rows × 3 columns (shorter, legible) with V's single shared legend, integer epoch ticks and model titles. Validation macro-F1 is dash-dot in the model colour and validation loss/accuracy black dashed. |
| V P2-13 in-figure titles duplicate captions | applied | Removed from Figs 5, 7, 8, B.1 and the galleries. |
| V P2-14 Fig 5 legend inside data | applied (= C P1-2) | Legend above the axes, light x-grid at 0.2 steps, `s=22` markers, no in-plot title. |
| V P2-15 Fig 3 repeated "no augmentation" | applied (= C P3-5) | One centred label across C's row; horizontal row labels at 7 pt. |
| V P2-16 sentence-initial "model C" | applied (= C P1-4) | `\MA/\MB/\MC`, used at sentence starts and in subcaptions. |
| V P2-17 decimal alignment | partially | Table B.3 (efficiency) uses `siunitx` S columns. Tables 4, A.1, A.2, B.2, B.4 and C.1 already have a fixed number of decimals per column, so `r`/`c` columns align and they were left as they are. |
| V P2-18 Table 4 CI and reference rows | applied | Small grey CI rows in brackets, grey reference rows, `\addlinespace` between model blocks. S columns not used (see P2-17). |
| V P2-19 Table 3 dense | applied | 3.1 cm first column (3.4 cm overflowed at the final font), five small-caps groups, `\arraystretch` line removed. |
| V P2-20 p.7 gap, Table 3 drift | applied | Resolved by the reflow; Table 3 now fills its page directly after Fig 3. |
| V P2-21 references: IEEEtranN + small font | partially | `\bibfont{\small}` and `\bibsep` applied. **Not** switched to IEEEtranN, because the user asked for `ieeetr`. The `.”` artefact is now confined to the three remaining `@misc` entries (Kaggle, timm, uv), since the arXiv `@misc` entries became published `@inproceedings` (C P2-5a). |
| V P3-1 wider margins | not applied | It would overflow six tables and add body length; there is no page slack. |
| V P3-2 running header | applied | `fancyhdr`, with the section name on the right (`\rightmark`, so subsections do not overwrite it). |
| V P3-3 "Key results" box | not applied | Costs about 5 lines with no page slack. The new "Main findings" list in §1 (C P1-3) carries the same content. |
| V P3-4 heading spacing / colour | applied | `titlesec` spacing as suggested; navy headings. |
| V P3-5 rubric longtable | applied | `\footnotesize`, `\linewidth`-based columns, "(continued)" header (C P3-7); now 2 pages. |
| V P3-6 Table 4 (environment) half height | applied differently | The four-column version was tried and was cramped (the long GPU/CPU strings wrapped to 7 lines). The table moved to Appendix B (Table B.1) instead; §4.4 still states the GPU, its memory and the peak VRAM, and points to it. |
| V P3-7 long captions | applied | Table 1 caption shortened with a column-legend note; the Table 6 summed-confusion sentence moved into §6.2 text. Several figure and table captions tightened. |
| V P3-8 Fig 8 cryptic labels | applied (= C P2-17) | Horizontal bars with plain labels ("all three", "B and C", …, "none"); the caption no longer needs a key. |
| V P3-9 microtype tuning | applied | `tracking=true` with small-caps tracking 40; protrusion and expansion are microtype's pdfTeX defaults. |
| V P3-10 class order by visual family | not applied | The review marked it optional, and it would change the reading order of Tables 1 and A.1–A.2. |
| V "do not change" list | kept | 9-part structure, macro pipeline, booktabs, Okabe–Ito palette, Blues matrices, bold best values, Table 5, run-in heads, Appendix D/E, cleveref and natbib options. |

## Content review (C)

| Item | Status | What was done / why not |
|---|---|---|
| C P1-1 figure text size | applied | = V P1-1. |
| C P1-2 Fig 5 legend | applied | = V P2-14. |
| C P1-3 research question + main findings | applied | Inserted after "Aim and scope", macros only. Wording: "data quality, not class imbalance, limits accuracy". |
| C P1-4 lower-case "model C" | applied | = V P2-16. |
| C P1-5 fairness paragraph | applied | §4.3 now says the benchmark used the 2,512 development images only and the test images were never used. New **Fairness** paragraph in §4.4 (the epoch budgets use `\maxEpochsA`/`\maxEpochsB`); subsection renamed "Software, hardware and reproducibility"; rubric rows updated. |
| C P1-6 two leakage statistics | applied | §2.5 adds "(counted over the raw file entries)"; Appendix C states the deduplicated-development-image denominator and points back to §2.5. |
| C P1-7 doubled spaces | applied | = V P2-3. |
| C P1-8 group-member placeholders | user to do | The placeholders remain (owed by the user; also the university/faculty line). |
| C P2-1 one name per model | applied | `SHORT` in `make_tables.py` and `itri626_helpers.py`: "A: scratch CNN", "B: ConvNeXt-T (fine-tuned)", "C: SigLIP ViT (frozen + linear)". Prose uses "VGG-style CNN" and "model A/B/C"; "FT" and "custom CNN" are removed except in the brief's own wording ("small custom CNN"). |
| C P2-2 split procedure | applied | Rewritten as proposed; facts verified in `tools/build_dataset_v2.py` (7 folds, min total-variation distance, inner 5 folds, seed 42). |
| C P2-3 abstract ≤ 200 words | applied | The reviewer's 191-word version, macros only. |
| C P2-4 appendix refs / floats / numbering | partially | (1) applied via `\crefalias`; (3) applied: `\counterwithin` makes Table A.1 … E.1 and Figure B.1. (2) applied to the appendices via `\FloatBarrier`; **not** as `placeins[section]` in the body, because forcing the §5 floats before §6 cost about half a page, which the 15-page body cannot spare. `flafter` keeps every float after its first citation. |
| C P2-5a arXiv → published versions | applied | Verified on the publisher pages: VGG (ICLR 2015), BatchNorm (PMLR 37:448–456), ViT (ICLR 2021, OpenReview), CLIP (PMLR 139:8748–8763), Kumar et al. (ICLR 2022, OpenReview), AdamW (ICLR 2019, OpenReview), SGDR (ICLR 2017, OpenReview), Geifman & El-Yaniv (NeurIPS 30, proceedings.neurips.cc). Also verified via CrossRef: Shao et al. [1] (previously arXiv 2025) is published as Proc. 7th Int. Conf. Universal Village (UV) 2024, pp. 1–10, DOI 10.1109/uv63228.2024.11189118, so the entry was updated. |
| C P2-5b Kaggle n.d., no access date | applied | `title = {Electronic Component Recognizer}`, `howpublished = Kaggle dataset (20 classes), URL`, `note = Accessed 17 October 2026` (manifest `created_utc` 2026-10-17). |
| C P2-5c `ieeetr` misc punctuation | partially | Style kept as `ieeetr` (user constraint). Only three `@misc` entries remain. |
| C P2-5d title case | applied | All cited entries in sentence case with protected acronyms. |
| C P2-5e venue names | applied | Abbreviated IEEE forms throughout (NeurIPS, IJCV, J. Mach. Learn. Res., …). |
| C P2-6 Fig 2 untidy | applied | Superseded by V P1-5 (the redesign was chosen over C's in-place fix). |
| C P2-7 jargon | applied | All nine: linear probe defined at first use (with `\label{sec:modelC}`), CLIP cited, pHash explained, "OOD" → "flagged" / "off-topic flag" and spelled out in the Table 1 note, McNemar explained, WebLI explained, crop fraction instead of `crop_pct`, zero-shot text head explained, drop-path rate, "CE: cross-entropy" in Tables B.2 and B.4. |
| C P2-8 conclusion answers the question + lessons | applied | Opening sentence, R2 sentence and a **Lessons learned** paragraph (macros only). |
| C P2-9 evidence for §6.1 claims | applied | Zero-shot macro-F1 (`\bmZeroSiglip`) and flagged-image accuracies (`\accOODC`, `\accOODB`); the A-confusion sentence is hedged. |
| C P2-10 limitations | applied | Two new items (screen/model lineage, residual near-duplicates); test-label point merged into "Noisy labels". |
| C P2-11 file paths in prose | applied | `url` breaks (see V P2-4). Six result-file paths moved from the prose to a "Provenance" paragraph in Appendix D; the captions keep their sources. |
| C P2-12 number formatting | partially | (a) thousands separators for `vramB` and `fpMBC`: applied. (b) Table 5 caption and "percentage points": applied. (c) `.3g` in Table B.3: **not applied**. It would re-round reported values (e.g. 893.5 → 894). Decimal alignment (V P2-17) addresses the readability instead, so this resolves the conflict in V's favour. (d) "5-fold": applied. (e) "The 3 smallest values (10⁻⁴ to 10⁻²)": applied (`\nTies`). |
| C P2-13 original split mentioned before use | applied | Sentence added to §2.1. |
| C P2-14 learning curves | applied | = V P2-12. |
| C P2-15 galleries | applied | = V P1-2 / P2-11. |
| C P2-16 disagreement layout | applied | 3 rows × 4 with row labels; each title gives the true class (bold) and A/B/C predictions with ✓/✗; caption rewritten. |
| C P2-17 Fig 8 labels | applied | = V P3-8. |
| C P2-18 confusion colour bars | applied | = V P1-3. |
| C P2-19 Table 2 wording | applied | "Final images", "images in near-duplicate clusters", "split units: … (groups, not images)", "images flagged as possibly off-topic"; caption shortened. |
| C P2-20 §3.4 heading | applied | "Rationale for the models and evaluation protocol" plus the sentence on the brief's families. |
| C P2-21 R2 is not a fourth model | applied | Sentence added in §5.1. |
| C P3-1 title | applied | = V P1-8. "Facilitator" kept singular (one name). |
| C P3-2 elliptical §4.3 sentence | applied | Split into three sentences. |
| C P3-3 diode→led "2 times" | applied | "(2 of C's errors are diode→led)" (macro `\cmDiodeLedC`). |
| C P3-4 log-loss | applied | One sentence with `\loglossC`. |
| C P3-5 Fig 3 | applied | = V P2-15. |
| C P3-6 Fig B.1 | applied | Panel titles removed (they caused the rounding mismatch); legend "final (after cleaning)". |
| C P3-7 Table E.1 | applied | "(continued)" header, "(brief §…)", notebook-figure row replaced by report sections. |
| C P3-8 ethics: representativeness | applied | Sentence added. The thumbnail note was already covered by the existing "small thumbnails from the public dataset" sentence. |
| C P3-9 §6.3 heading | applied | "Supporting evidence from the wider study". |
| C P3-10 paragraph spacing | applied | = V P2-2. |
| C P3-11 links | applied | = V P2-5. |
| C P3-12 "weighted averages" | applied | "macro averages and weighted F1". |
| C P3-13 Fig 7 caption | applied | Confidence defined and abbreviations listed; "true → predicted" explained. |
| C "do not change" list | kept | Number pipeline, 9-part structure, run-in heads, model colours, Table 5 and single test use, provenance statements, British spelling, sources in captions, Appendices D/E, numeric citation phrasing. |

## Files changed

- `main.tex`: preamble, title page, text edits above, float layout.
- `make_tables.py`: model names, colour keys, Table 1/2/3/4(env)/5/6/B.3 layout, thousands separators. Regenerated `generated/*.tex`, `numbers.tex` and `key_numbers.json`.
- `refs.bib`, `extra.bib`: cited entries normalised and published versions, Kaggle access date.
- `RUBRIC_CHECKLIST.md`: table, figure and section numbers updated to this build.
- `notebooks/itri626_helpers.py`: figure functions (see below); `results/itri626/figures/*.pdf|png` regenerated locally from the saved outputs. No network was loaded; Fig 3 uses only the torchvision transforms.
- New figure `results/itri626/figures/fig06_confusion_grid.{pdf,png}`.

### Helper functions changed (re-run the notebook's figure cells on the GPU PC to refresh its embedded figures)

`apply_figure_style` (default sizes), `SHORT`, `WRONG`, plus the new `OK`, `TEXTW`, `ABBR`/`short_cls`, `_frame`,
`_two_line`, `ROWLABEL`, `_draw_cm`, `DISAGREE_ROWS` and `AGREE_LABEL`. Rewritten or changed: `fig_class_counts`,
`fig_examples`, `fig_augmentation`, `fig_learning_curves`, `fig_test_metrics`, `fig_confusion` (new optional
`vmax=`), **new** `fig_confusion_grid`, `fig_per_class_f1`, `fig_gallery` (`title` now optional and not drawn),
`fig_disagreement` (default `ncol=4`), `fig_agreement`, `fig_efficiency` and `fig_grouped`. The notebook cells keep
working unchanged. Optional notebook updates: pass `vmax=` to `fig_confusion` for a shared scale, and add
`H.save(H.fig_confusion_grid({n: metrics["models"][n]["test"]["confusion"] for n in H.MAIN}, classes), "fig06_confusion_grid")`
so the notebook also writes the grid that the report uses.


# Revision 2026-11-04: model A trained to convergence; improvement study added

- `results/itri626/`: model A's run replaced by the 200-epoch recipe (`train.epochs=200 train.patience=40`, config
  hash `b0afa2aa4d60`), imported from the improvement study's fold-0 run with `tools/itri626_import_run.py` (no
  retraining; the sealed test was scored once for that recipe, see `EXPERIMENT_LOG.md` 2026-11-04). Test accuracy /
  macro-F1 0.354 / 0.336 -> 0.400 / 0.392. B, C and the reference rows are unchanged. `scripts/itri626_experiments.py`
  now carries the same `--set` overrides for A. `step collect` and `make_tables.py` re-run; 93 number macros changed.
- `main.tex`: hyperparameters, fairness, software/hardware (A's final run was on a second machine), learning-curve
  bullet A, "why the models differ", conclusion, reproduction appendix and the environment-table caption updated; new
  subsection "Can the scratch CNN be improved?" (`sec:improveA`) with `generated/tab_improveA.tex`.
- `notebooks/ITRI626_four_models.ipynb`: numbers in sections 7 that quoted A updated; new section 8 built from the
  saved files; notebook re-executed on the laptop (figures regenerated from the saved outputs).
- `README.md`: model A row of the submission table.


# Revision 2026-11-05: PR review corrections

- Regenerated `results/itri626/figures/fig06_confusion_grid.{pdf,png}` from the saved A/B/C confusion matrices.
  The report grid had retained the 120-epoch A run (148 correct predictions); it now matches the 200-epoch
  run (167 correct, including 13 battery holders). Rebuilt `main.pdf`; no model was loaded or scored.
- The documented CPU reproduction command keeps R3 metadata when refreshing R1/R2. The training notebook
  builds the four original app manifests and copies metadata for Small and crop 1.0 without requiring their graphs.
- Dataset rebuilds without the original split file preserve recorded legacy split-leakage flags and matching
  historical audit counts. Missing audit evidence is marked unavailable rather than counted as zero.

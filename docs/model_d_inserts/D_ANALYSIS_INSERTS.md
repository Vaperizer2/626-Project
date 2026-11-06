# Hand-over: add model D to the analysis passages

These changes add D wherever the results and discussion cover only A, B and C. On the `Meiring` branch they have
been applied to `docs/itri626_report/`; this file records them for review. No model was retrained or rerun: all
numbers come from the recorded files
`results/itri626/metrics.json`, `results/itri626/per_class.csv` and `results/itri626/preds/*_test.npz`.

**Page check.** A trial build with all six changes on a scratch copy of the report's committed source still ends the
Conclusion on PDF page 15 (printed page 14), the same page as now. References still start on that page, with about a
fifth of a page left. The total grows from 30 to 31 PDF pages only because of the new appendix figure. The build had
no overfull boxes. Any other edits made to the report since will use some of that space, so rebuild to confirm.

## 1. Section 5.3, per-class results

After "...(microcontroller board, switch, jumper wire, heat sink)." add:

```latex
D is weakest on \cls{led} (F1 0.61) and strongest on \cls{potentiometer} (0.93).
```

## 2. Section 5.5, agreement between the models

After "...(examples in Figure 7)." add:

```latex
D recognises 14 of the 57 images that A, B and C all miss, leaving 43 that no model gets right, and is wrong on 27
images that A or B get right.
```

## 3. Section 6.1, why the models differ

In the Spearman bracket, replace "+0.56 for B and +0.43 for C" (`\rhoB for B and \rhoC for C`) with:

```latex
\rhoB for B, \rhoC for C and +0.20 for D
```

## 4. Section 6.2, look-alike components

Replace the last sentence, "Summed over A, B and C, the most frequent confusions are ...", with:

```latex
D's largest is \cls{ic chip} $\rightarrow$ \cls{microcontroller board} (4), which it shares with B, and it shares
\cls{jumper wire} $\rightarrow$ \cls{connector} (3) with C. Summed over all four models, the most frequent confusions
are \cls{jumper wire} $\rightarrow$ \cls{connector} (10), \cls{transistor} $\rightarrow$ \cls{diode} (10) and
\cls{resistor} $\rightarrow$ \cls{crystal oscillator} (9).
```

Then update the table:

- Use `docs/model_d_inserts/tables/tab_confused_four_models.tex` in place of `generated/tab_confused.tex`. Note that
  `make_tables.py` regenerates the old three-model file, so `\input` the new file under its own name.
- Change that table's `\centering\footnotesize` to `\centering\scriptsize`. At footnote size the four columns are
  45 pt too wide for the page.
- Replace the caption with:

```latex
\caption{Most frequent test confusions (true $\rightarrow$ predicted) per model, from their recorded confusion
matrices; equal counts are listed alphabetically. MCU board: microcontroller board; crystal osc.: crystal oscillator;
PSU module: power supply module.}
```

## 5. Section 6.2, confidence is informative

Replace "(B: 0.76 and 0.43)" (`(B: \confOkB and \confBadB)`) with:

```latex
(B: \confOkB and \confBadB; D: 0.90 and 0.56)
```

## 6. Appendix A, new figure

Copy `docs/model_d_inserts/figures/fig09_agreement_four_models.pdf` next to `main.tex` and add this directly after the
existing agreement figure (`fig:agree`). Keep that figure, because the counts in section 5.5 (143, 109, 75, 57) are
read from it.

```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=0.5\textwidth]{fig09_agreement_four_models.pdf}
\caption{Test images by the subset of all four models that classifies them correctly.}
\label{fig:agreeD}
\end{figure}
```

## Left as A, B and C

The disagreement figure in section 5.5 (`fig:disagree`) stays as it is. Adding D means a fourth prediction line in each
panel and a new image selection, which costs body space, and D's own correct and incorrect predictions are already
shown in `fig:galleryD`.

# Recognising electronic components in web photographs

*{{SUBTITLE}}*

ITRI626, Deep learning for image classification, practical project report. {{DATES}}.
Markdown edition with model D, prepared by Meiring van Niekerk for the team lead.

> **How to read this file.** It is the course report (`docs/itri626_report/main.tex`) converted to Markdown, with
> model D written in wherever it belongs. <ins>Underlined text is added</ins> and <del>struck-through text is
> removed</del>, both relative to the report as it stands on branch `Meiring`. A note in a quote block like this one
> marks every table or figure that is new or changed. Unmarked text is the report's own, unchanged.
>
> Nothing under `docs/itri626_report/` was edited to make this file. The same additions are listed one by one, with
> their locations and a core or optional mark, in `INSERTS.md` next to this file; the tables are available as LaTeX
> in `tables/`, and the four-model figures as PDF in `results/itri626/figures_four_models/`.
>
> **Three points for the team lead.** (1) The convolution-attention hybrid needs the lecturer's approval as a fourth
> family under "another approved image classification architecture"; A, B and C satisfy the brief without it.
> (2) Page budget: this version was built in LaTeX and its body is 14.8 pages, the same as the report today, provided
> the hyperparameter table (Table 3, now four columns) moves to Appendix B; left in the body it is 15.8. It is shown
> in its present place here only so that the table numbers match the report. (3) Section 5.1 can no longer say that
> all 95 % bootstrap intervals are disjoint: those of C and D overlap and the paired test separates them.
>
> **One proposed cut, to make room.** The distillation experiment on model A is removed: its sentences leave
> Section 6.3, its row leaves Table 7, its clause leaves the conclusion and its reference leaves the list. This is a
> suggestion, marked like the other changes; keep the original text if you prefer.

**Abstract.** {{ABSTRACT}}

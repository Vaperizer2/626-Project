"""Render the report's TikZ workflow diagram (Figure 2) as a PNG for the Markdown edition.

    python docs/model_d_inserts/build_workflow_figure.py REPORT_DIR OUT.png

REPORT_DIR is a checkout of docs/itri626_report/ (read only); needs pdflatex and pdftoppm on PATH.
"""
import re
import subprocess
import sys
from pathlib import Path

d, out = Path(sys.argv[1]), Path(sys.argv[2])
tex = (d / "main.tex").read_text(encoding="utf-8")
pic = re.search(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", tex, flags=re.S).group(0)
colours = "\n".join(re.findall(r"\\definecolor\{m[A-D]\}\{HTML\}\{[0-9A-Fa-f]+\}", tex))
doc = "\n".join([
    r"\documentclass[border=6pt]{standalone}",
    r"\usepackage[T1]{fontenc}\usepackage{XCharter}\usepackage[xcharter]{newtxmath}",
    r"\usepackage{xcolor}\usepackage{xspace}\usepackage{tikz}",
    r"\usetikzlibrary{arrows.meta,positioning,calc}",
    colours,
    r"\input{generated/numbers.tex}",
    r"\begin{document}",
    pic,
    r"\end{document}",
])
(d / "_workflow.tex").write_text(doc, encoding="utf-8")
subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "_workflow.tex"], cwd=d, check=True,
               stdout=subprocess.DEVNULL)
subprocess.run(["pdftoppm", "-r", "200", "-png", "-singlefile", "_workflow.pdf", str(out.with_suffix(""))], cwd=d, check=True)
print("wrote", out)

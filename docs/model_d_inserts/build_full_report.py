"""Build FULL_REPORT_WITH_MODEL_D.md: the course report as Markdown with model D added and the changes marked.

    python docs/model_d_inserts/build_full_report.py BASE_DIR DRAFT_DIR OUT_DIR

BASE_DIR and DRAFT_DIR are two checkouts of docs/itri626_report/ that have been built with latexmk (main.aux gives
the section, table and figure numbers): BASE = the report on branch Meiring (three models), DRAFT = the trial build
with model D from branch Meiring-four-model-draft. The report itself is only read. Needs pandoc on PATH and a CSL file
(ieee.csl) next to BASE_DIR's parent; see the commands in EXPERIMENT_LOG.md (2026-11-05).

Steps per version: expand the number macros from generated/key_numbers.json, resolve \\cref from main.aux, inline the
generated tables with plain column specifications, flatten the composite figures, number the captions, convert with
pandoc (GitHub Markdown, IEEE citations), number the headings. Then the two Markdown texts are aligned block by block
and the DRAFT is written with additions in <ins> and removals in <del>; changed or new tables and figures get a note.
"""
import difflib
import json
import re
import subprocess
import sys
from pathlib import Path

LAB = re.compile(r"\\newlabel\{([^}@]+)@cref\}\{\{\[(\w+)\]\[[^\]]*\]\[[^\]]*\]([^}]*)\}")
KIND = {"section": "Section", "subsection": "Section", "table": "Table", "figure": "Figure", "appendix": "Appendix",
        "subappendix": "Appendix", "enumi": "item", "subfigure": "Figure"}
PLURAL = {"Section": "Sections", "Table": "Tables", "Figure": "Figures", "Appendix": "Appendices", "item": "items"}


def labels(d: Path) -> dict:
    t = (d / "main.aux").read_text(encoding="utf-8", errors="replace")
    return {m.group(1): (KIND.get(m.group(2), m.group(2).capitalize()), m.group(3)) for m in LAB.finditer(t)}


def cref(keys: str, lab: dict) -> str:
    items = [lab[k.strip()] for k in keys.split(",")]
    kinds = {k for k, _ in items}
    if len(items) == 1:
        return f"{items[0][0]} {items[0][1]}"
    if len(kinds) == 1:
        nums = [n for _, n in items]
        return f"{PLURAL[items[0][0]]} {', '.join(nums[:-1])} and {nums[-1]}"
    return ", ".join(f"{k} {n}" for k, n in items[:-1]) + f" and {items[-1][0]} {items[-1][1]}"


def expand_macros(s: str, nums: dict) -> str:
    fixed = {"mA": "model A", "mB": "model B", "mC": "model C", "mD": "model D",
             "MA": "Model A", "MB": "Model B", "MC": "Model C", "MD": "Model D"}
    table = {**{k: v["value"] for k, v in nums.items()}, **fixed}
    names = sorted(table, key=len, reverse=True)
    pat = re.compile(r"\\(" + "|".join(map(re.escape, names)) + r")(?![A-Za-z])(\{\})?[ \t]*(\n?)")

    def sub(m):
        val = table[m.group(1)]
        rest = s[m.end():m.end() + 2]
        nl = m.group(3)
        if m.group(2):  # \name{} : explicit end, keep the following space as written
            return val + (" " if s[m.end() - 1:m.end()] in " \t" else "") + nl
        nospace = rest[:1] in ",.'/?;:!~-)]}$" or rest[:2] in ("\\,", "\\%") or rest[:1] == ""
        if nl:
            return val + nl
        return val + ("" if nospace else " ")

    return pat.sub(sub, s)


def fix_file(m) -> str:
    inner = m.group(1).replace("\\_", "_")
    return "\\texttt{" + inner.replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("%", "\\%").replace("#", "\\#").replace("&", "\\&") + "}"


def simple_colspec(spec: str) -> str:
    """A tabular column specification reduced to l/c/r letters (pandoc needs nothing more)."""
    spec = re.sub(r"\*\{(\d+)\}\{([^{}]*)\}", lambda m: m.group(2) * int(m.group(1)), spec)
    spec = re.sub(r"@\{[^{}]*\}", "", spec)
    spec = re.sub(r">\{(?:[^{}]|\{[^{}]*\})*\}", "", spec)
    spec = re.sub(r"S\[[^\]]*\]", "r", spec)
    spec = re.sub(r"p\{(?:[^{}]|\{[^{}]*\})*\}", "l", spec)
    spec = spec.replace("Y", "l").replace("X", "l").replace("S", "r").replace("|", "")
    return re.sub(r"[^lcr]", "", spec)


def balanced(s: str, i: int) -> int:
    """Index just after the group that opens at s[i] == '{'."""
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return j + 1
    raise ValueError("unbalanced")


def norm_tables(s: str) -> str:
    s = re.sub(r"\\endfirsthead.*?\\endhead", "", s, flags=re.S)  # longtable: keep the first head only
    s = re.sub(r"\\bottomrule\s*\\endfoot", "", s)
    out, i = [], 0
    pat = re.compile(r"\\begin\{(tabularx|tabular|longtable)\}")
    while True:
        m = pat.search(s, i)
        if not m:
            out.append(s[i:])
            break
        out.append(s[i:m.start()])
        j = m.end()
        if m.group(1) == "tabularx":  # {width}{spec}
            j = balanced(s, j)
        k = balanced(s, j)
        out.append("\\begin{" + ("tabular" if m.group(1) != "longtable" else "longtable") + "}{" + simple_colspec(s[j + 1:k - 1]) + "}")
        i = k
    s = "".join(out).replace("\\end{tabularx}", "\\end{tabular}")
    s = re.sub(r"\\cmidrule(\([^)]*\))?\{[^}]*\}", "", s)
    s = re.sub(r"\\addlinespace(\[[^\]]*\])?", "", s)
    s = re.sub(r"\\msq\{[A-D]\}", "", s)
    s = re.sub(r"\{\\(?:scriptsize|footnotesize|small)\\color\{[^}]*\}", "{", s)
    s = re.sub(r"\\color\{[^}]*\}", "", s)
    s = re.sub(r"\\num\{([^}]*)\}", lambda m: m.group(1), s)
    s = s.replace("\\multicolumn{8}{@{}l}", "\\multicolumn{8}{l}").replace("\\multicolumn{5}{@{}l}", "\\multicolumn{5}{l}")
    s = s.replace("\\multicolumn{4}{@{}l}", "\\multicolumn{4}{l}").replace("\\multicolumn{2}{@{}l}", "\\multicolumn{2}{l}")
    return s


def number_captions(s: str, lab: dict) -> str:
    def float_block(m):
        blk = m.group(0)
        kind = "Table" if m.group(1) == "table" else "Figure"
        caps = list(re.finditer(r"\\caption\{", blk))
        labs = list(re.finditer(r"\\label\{((?:tab|fig):[^}]+)\}", blk))
        for cm in reversed(caps):  # each caption takes the first label after it
            after = [l for l in labs if l.start() > cm.start()]
            if not after:
                continue
            num = lab[after[0].group(1)][1]
            blk = blk[:cm.end()] + f"\\textbf{{{kind} {num}.}} " + blk[cm.end():]
        return blk

    s = re.sub(r"\\begin\{(table|figure)\}.*?\\end\{\1\}", float_block, s, flags=re.S)
    s = s.replace("\\caption{Brief requirement", f"\\caption{{\\textbf{{Table {lab['tab:rubric'][1]}.}} Brief requirement")
    return s


def flatten_figures(s: str) -> str:
    # the TikZ workflow diagram becomes an image
    s = re.sub(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", r"\\includegraphics{fig00_workflow.pdf}", s, flags=re.S)
    # subfigures: one image after the other, sub-captions as plain lines
    s = re.sub(r"\\begin\{subfigure\}\{[^}]*\}\\centering\s*(\\includegraphics(?:\[[^\]]*\])?\{[^}]*\})\s*\\caption\{([^}]*)\}\s*\\end\{subfigure\}",
               lambda m: m.group(1) + "\n\n" + "\\emph{" + m.group(2) + "}\n", s)
    # the agreement / disagreement pair set with minipages: two ordinary figures
    m = re.search(r"\\begin\{figure\}\[tb\]\s*\\centering\s*\\begin\{minipage\}.*?\\end\{figure\}", s, flags=re.S)
    if m:
        blk = m.group(0)
        imgs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", blk)
        caps = [(c.group(1), l.group(1)) for c, l in zip(re.finditer(r"\\caption\{((?:[^{}]|\{[^{}]*\})*)\}", blk), re.finditer(r"\\label\{([^}]*)\}", blk))]
        new = "".join(f"\\begin{{figure}}\n\\includegraphics{{{img}}}\n\\caption{{{cap}}}\n\\label{{{lb}}}\n\\end{{figure}}\n\n"
                      for img, (cap, lb) in zip(imgs, caps))
        s = s.replace(blk, new)
    # two images in one figure with two captions (per-class F1 above the confusion grid): split
    m = re.search(r"\\begin\{figure\}\[p\]\s*\\centering\s*\\includegraphics\[width=0\.8\\textwidth\]\{fig07_per_class_f1\.pdf\}.*?\\end\{figure\}", s, flags=re.S)
    if m:
        blk = m.group(0)
        a, b = blk.split("\\medskip")
        s = s.replace(blk, a.rstrip() + "\n\\end{figure}\n\n\\begin{figure}[p]\n\\centering" + b)
    return s


def to_markdown(d: Path, csl: Path) -> tuple[str, str]:
    lab = labels(d)
    nums = json.loads((d / "generated" / "key_numbers.json").read_text(encoding="utf-8"))
    tex = (d / "main.tex").read_text(encoding="utf-8")
    abstract = re.search(r"\\textbf\{\\color\{accent\}Abstract\.\}(.*?)\\end\{minipage\}", tex, flags=re.S).group(1)
    subtitle = re.search(r"\{\\large (A leakage-free comparison.*?)\\par\}", tex, flags=re.S).group(1)
    dates = re.search(r"(\d+ October 2026, revised \d+ November 2026)", tex).group(1)
    body = tex[tex.index("\\section{Introduction"):tex.index("\\end{document}")]
    body = body.replace("\\appendix", "\n\nAPPENDIXSTARTSHERE\n\n")
    body = re.sub(r"\\bibliographystyle\{[^}]*\}\s*\\bibliography\{[^}]*\}", "\n\nREFERENCESGOHERE\n\n", body)
    body = re.sub(r"\\input\{(generated/[^}]+)\}", lambda m: re.sub(r"^%.*\n", "", (d / m.group(1)).read_text(encoding="utf-8")), body)
    parts = {}
    for name, s in (("body", body), ("abstract", abstract), ("subtitle", subtitle)):
        s = re.sub(r"(?<!\\)%.*", "", s)                      # comments
        # each citation keeps its keys in a marker, so both versions can be numbered alike afterwards (recite)
        s = re.sub(r"\\citep\{([^}]*)\}", lambda m: m.group(0) + "CITEKEYS" + m.group(1).replace(" ", "").encode().hex() + "ENDCITE", s)
        s = flatten_figures(s)
        s = number_captions(s, lab)
        s = re.sub(r"\\[cC]ref\{([^}]*)\}", lambda m: cref(m.group(1), lab), s)
        s = expand_macros(s, nums)
        s = re.sub(r"\\file\{([^{}]*)\}", fix_file, s)
        s = re.sub(r"\\cls\{([^{}]*)\}", r"\\emph{\1}", s)
        s = norm_tables(s)
        s = re.sub(r"\\label\{[^}]*\}", "", s)
        for junk in ("\\FloatBarrier", "\\clearpage", "\\medskip", "\\smallskip", "\\centering", "\\footnotesize", "\\scriptsize",
                     "\\crefalias{section}{appendix}", "\\counterwithin{figure}{section}\\counterwithin{table}{section}",
                     "\\renewcommand{\\bibfont}{\\small}\\setlength{\\bibsep}{2pt plus 1pt}"):
            s = s.replace(junk, "")
        s = re.sub(r"\\setlength\{\\tabcolsep\}\{[^}]*\}", "", s)
        s = re.sub(r"\\begin\{(itemize|enumerate)\}\[[^\]]*\]", r"\\begin{\1}", s)
        s = s.replace("\\\\[4pt]", " ").replace("\\,", " ")
        parts[name] = s
    src = ("\\documentclass{article}\\begin{document}\n" + "ABSTRACTSTART\n\n" + parts["abstract"] + "\n\nABSTRACTEND\n\n"
           + "SUBTITLESTART\n\n" + parts["subtitle"].replace("\\\\", " ") + "\n\nSUBTITLEEND\n\n" + parts["body"] + "\n\\end{document}\n")
    (d / "_md_source.tex").write_text(src, encoding="utf-8")
    r = subprocess.run(["pandoc", "-f", "latex", "-t", "gfm", "--wrap=none", "--citeproc", "--bibliography", "refs.bib",
                        "--bibliography", "extra.bib", "--csl", str(csl), "_md_source.tex"], cwd=d, capture_output=True, text=True,
                       encoding="utf-8")
    if r.returncode != 0:
        raise SystemExit("pandoc failed: " + r.stderr[-2000:])
    if r.stderr.strip():
        print("pandoc warnings:", r.stderr.strip()[:1500], file=sys.stderr)
    return r.stdout, dates


def number_headings(md: str) -> str:
    out, sec, sub, app = [], 0, 0, False
    for ln in md.splitlines():
        if ln.strip() == "APPENDIXSTARTSHERE":
            app, sec, sub = True, 0, 0
            continue
        m = re.match(r"^(#{1,2}) (.*)$", ln)
        if m and not ln.startswith("# References"):
            if len(m.group(1)) == 1:
                sec, sub = sec + 1, 0
                tag = chr(64 + sec) if app else str(sec)
                ln = f"## {tag} {m.group(2)}"
            else:
                sub += 1
                tag = (chr(64 + sec) if app else str(sec)) + f".{sub}"
                ln = f"### {tag} {m.group(2)}"
        out.append(ln)
    return "\n".join(out)


def tidy(md: str) -> dict:
    """Split the pandoc output into abstract, subtitle, body (references moved before the appendices)."""
    md = md.replace("\r\n", "\n")
    abstract = md[md.index("ABSTRACTSTART") + 13:md.index("ABSTRACTEND")].strip()
    subtitle = md[md.index("SUBTITLESTART") + 13:md.index("SUBTITLEEND")].strip()
    body = md[md.index("SUBTITLEEND") + 11:].strip()
    refs = ""
    i = body.find('<div id="refs"')
    if i >= 0:
        refs, body = body[i:].strip(), body[:i].strip()
        refs = re.sub(r"</?div[^>]*>", "", refs)
        refs = re.sub(r'<span class="csl-left-margin">(.*?)</span><span class="csl-right-inline">(.*?)</span>', r"\1 \2", refs, flags=re.S)
        refs = re.sub(r"</?span[^>]*>", "", refs)
        refs = "\n\n".join(p.strip() for p in refs.split("\n\n") if p.strip())
    body = number_headings(body)
    body = body.replace("REFERENCESGOHERE", "## References\n\n" + refs)
    body = re.sub(r'(src|data)="([^"/]+)\.pdf"', r'\1="figures/\2.png"', body)
    body = re.sub(r"\]\(([^)/]+)\.pdf\)", r"](figures/\1.png)", body)
    body = re.sub(r'<embed src="(figures/[^"]+)"[^>]*/>', r'<img src="\1" alt="" />', body)
    body = re.sub(r"</?span[^>]*>", "", body).replace("<!-- -->", "")  # pandoc's math / small-caps / colour wrappers
    body = re.sub(r"^</?div[^>]*>\n?", "", body, flags=re.M)        # size wrappers around the command block
    body = re.sub(r"\n{3,}", "\n\n", body)
    return {"abstract": demath(abstract), "subtitle": demath(subtitle), "body": demath(body)}


SUP = str.maketrans("0123456789-+", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺")
MATH = (("\\rightarrow", "→"), ("\\to", "→"), ("\\pm", "±"), ("\\times", "×"), ("\\Delta", "Δ"), ("\\rho", "ρ"),
        ("\\ln", "ln"), ("\\{", "{"), ("\\}", "}"), ("\\,", " "), ("\\cdot", "·"))


def demath(md: str) -> str:
    """Inline TeX math, which pandoc writes as $`...`$, as plain Unicode text (the report uses only simple formulas)."""
    def one(m):
        x = m.group(1).replace("^{\\ddagger}", "‡").replace("^{\\dagger}", "†")
        x = re.sub(r"\^\{([-+0-9]+)\}", lambda k: k.group(1).translate(SUP), x)
        for a, b in MATH:
            x = x.replace(a, b)
        return "−" if x.strip() == "-" else x

    md = re.sub(r"\$`([^`]*)`\$", one, md)
    return md.replace("p= ", "p = ").replace("ρ=+", "ρ = +")


CITE = re.compile(r"((?:\\\[\d+\\\](?:, |–|-)?)+)CITEKEYS([0-9a-f]+)ENDCITE")
REF = re.compile(r"^\\\[(\d+)\\\]\s+(.*)$", flags=re.S)


def cite_numbers(md: str) -> dict:
    """key -> number, in order of first citation (the IEEE rule), read from the markers."""
    cmap = {}
    for m in CITE.finditer(md):
        for k in bytes.fromhex(m.group(2)).decode().split(","):
            cmap.setdefault(k, len(cmap) + 1)
    return cmap


def recite(md: str, cmap: dict) -> str:
    return CITE.sub(lambda m: ", ".join(f"\\[{n}\\]" for n in sorted(cmap[k] for k in bytes.fromhex(m.group(2)).decode().split(","))), md)


def renumber_refs(base_body: str, draft_body: str) -> str:
    """Give the base version's reference entries the draft's numbers (same entry text), so only new entries differ."""
    num = {}
    for blk in draft_body.split("\n\n"):
        m = REF.match(blk)
        if m:
            num[m.group(2).strip()] = m.group(1)
    out = []
    for blk in base_body.split("\n\n"):
        m = REF.match(blk)
        if m and m.group(2).strip() in num:
            blk = f"\\[{num[m.group(2).strip()]}\\] {m.group(2)}"
        out.append(blk)
    return "\n\n".join(out)


# ------------------------------------------------------------------------------------------------ marking the changes
def kind(block: str) -> str:
    if block.startswith("    ") or block.startswith("```"):
        return "code"
    b = block.lstrip()
    if b.startswith("#"):
        return "heading"
    if b.startswith("|") or b.startswith("<table"):
        return "table"
    if b.startswith("<figure") or b.startswith("![") or b.startswith("<img"):
        return "figure"
    if b.startswith("```") or b.startswith("    "):
        return "code"
    return "prose"


def ok_run(run: str) -> bool:
    """An inserted or deleted run may be wrapped only if it does not split Markdown markup."""
    tags = re.findall(r"</?[a-zA-Z][^<>]*>", run)
    opens = sum(1 for t in tags if not t.startswith("</") and not t.endswith("/>"))
    closes = sum(1 for t in tags if t.startswith("</"))
    rest = re.sub(r"</?[a-zA-Z][^<>]*>", "", run)
    html_ok = opens == closes and not re.search(r"</?[a-zA-Z]", rest)
    return run.count("**") % 2 == 0 and run.count("`") % 2 == 0 and run.count("$") % 2 == 0 and "|" not in run \
        and (run.count("*") - 2 * run.count("**")) % 2 == 0 and run.count("[") == run.count("]") and html_ok


def word_diff(old: str, new: str) -> str | None:
    """`new` with additions in <ins> and removals in <del>, word by word; None if the markup cannot be kept intact."""
    a, b = re.findall(r"\S+", old), re.findall(r"\S+", new)
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    out = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            out.append(" ".join(b[j1:j2]))
            continue
        gone, come = " ".join(a[i1:i2]), " ".join(b[j1:j2])
        if (gone and not ok_run(gone)) or (come and not ok_run(come)):
            return None
        if gone:
            out.append(f"<del>{gone}</del>")
        if come:
            out.append(f"<ins>{come}</ins>")
    return " ".join(x for x in out if x)


def mark_lines(old: str, new: str) -> str:
    """Line-aware marking, so list markers and line structure survive."""
    ol, nl = old.split("\n"), new.split("\n")
    sm = difflib.SequenceMatcher(None, ol, nl, autojunk=False)
    out = []

    def split_marker(line):
        m = re.match(r"^(\s*(?:[-*]|\d+\.)\s+)(.*)$", line)
        return (m.group(1), m.group(2)) if m else ("", line)

    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            out += nl[j1:j2]
        elif op == "insert":
            for ln in nl[j1:j2]:
                mk, rest = split_marker(ln)
                out.append(mk + (f"<ins>{rest}</ins>" if rest.strip() and ok_run(rest) else rest + "  *(new)*"))
        elif op == "delete":
            for ln in ol[i1:i2]:
                mk, rest = split_marker(ln)
                if rest.strip() and ok_run(rest):
                    out.append(mk + f"<del>{rest}</del>")
                elif rest.strip():  # markup that a <del> would break: say it in words
                    out.append(mk + "**Removed:** " + rest)
        else:
            olds, news = ol[i1:i2], nl[j1:j2]
            for k, ln in enumerate(news):
                mk, rest = split_marker(ln)
                if k < len(olds):
                    _, orest = split_marker(olds[k])
                    wd = word_diff(orest, rest)
                    out.append(mk + (wd if wd is not None else rest + "  *(changed)*"))
                else:
                    out.append(mk + (f"<ins>{rest}</ins>" if rest.strip() and ok_run(rest) else rest + "  *(new)*"))
            for ln in olds[len(news):]:
                mk, rest = split_marker(ln)
                if rest.strip() and ok_run(rest):
                    out.append(mk + f"<del>{rest}</del>")
    return "\n".join(out)


FOUR_MODEL_FIGS = ("fig00_workflow", "fig04_learning_curves", "fig06_confusion_grid", "fig07_per_class_f1")  # fig03 stays the report's own
THRESH = {"prose": 0.55, "heading": 0.8, "table": 0.35, "figure": 0.5, "code": 0.5}


def untag(b: str) -> str:
    """A heading without its number, so a renumbered section still matches."""
    return re.sub(r"^#+\s+[A-E0-9.]+\s+", "", b) if b.startswith("#") else b


def similar(a: str, b: str, k: str) -> float:
    if k == "figure":
        fa, fb = re.findall(r"figures/[A-Za-z0-9_]+", a), re.findall(r"figures/[A-Za-z0-9_]+", b)
        return 1.0 if fa and fa == fb else 0.0
    if k == "table":  # the same table keeps its caption and first header cells
        a, b = a[:400], b[:400]
    return difflib.SequenceMatcher(None, untag(a), untag(b), autojunk=False).ratio()


def mark(old_md: str, new_md: str) -> tuple[str, dict]:
    A, B = old_md.split("\n\n"), new_md.split("\n\n")
    sm = difflib.SequenceMatcher(None, A, B, autojunk=False)
    out, stats = [], {"changed": 0, "new": 0, "removed": 0}
    NOTE_T = "> **Model D:** this table is changed (rows or columns for D added)."
    NOTE_TN = "> **Model D:** new table."
    NOTE_F4 = "> **Model D:** this figure is the four-model version; the report currently shows A, B and C."
    NOTE_FN = "> **Model D:** new figure."

    def gone(blocks):
        for ob in blocks:
            if kind(ob) == "prose" and ob.strip():
                stats["removed"] += 1
                # a whole block that the new version drops: struck through, or named in words if a <del> would break its markup
                one_line = len(ob.splitlines()) == 1
                out.append(f"<del>{ob}</del>" if ok_run(ob) and one_line else "**Removed:** " + ob)

    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            out += B[j1:j2]
            continue
        olds, news = A[i1:i2], B[j1:j2]
        o = 0  # olds before this index are consumed; pairing keeps the order of both texts
        for nb in news:
            k = kind(nb)
            best = None
            for oi in range(o, len(olds)):
                if kind(olds[oi]) != k:
                    continue
                r = similar(olds[oi], nb, k)
                if r >= THRESH[k] and (best is None or r > best[0]):
                    best = (r, oi)
            if best is None:
                stats["new"] += 1
                if k == "prose":
                    out.append(mark_lines("", nb))
                elif k == "heading":
                    out.append(nb + " *(new, model D)*")
                elif k == "table":
                    out += [NOTE_TN, nb]
                elif k == "figure":
                    out += [NOTE_FN, nb]
                else:
                    out.append(nb)
                continue
            gone(olds[o:best[1]])
            ob, o = olds[best[1]], best[1] + 1
            if ob == nb:
                out.append(nb)
                continue
            stats["changed"] += 1
            if k == "prose":
                out.append(mark_lines(ob, nb))
            elif k == "table":
                cut = "distillation from C" in ob and "distillation from C" not in nb
                out += ['> **Proposed cut:** the row "A + distillation from C" is removed from this table.' if cut else NOTE_T, nb]
            elif k == "figure":
                out.append(mark_figure(ob, nb))
            elif k == "heading":
                out.append(nb if untag(ob) == untag(nb) else nb + " *(changed)*")
            else:  # a command block: marks would show as literal text, so say it in a note
                out += ["> **Model D:** the command block below is changed (model D added).", nb]
        gone(olds[o:])
    # figures whose image changes with D get a note, whether or not their caption text changed
    final = []
    for blk in out:
        if kind(blk) == "figure" and any(f in blk for f in FOUR_MODEL_FIGS):
            final.append(NOTE_F4)
        final.append(blk)
    return "\n\n".join(final), stats


def mark_figure(old: str, new: str) -> str:
    """A figure whose caption changed: the caption text is word-diffed, the image kept."""
    mo, mn = re.search(r"<figcaption>(.*?)</figcaption>", old, flags=re.S), re.search(r"<figcaption>(.*?)</figcaption>", new, flags=re.S)
    if mo and mn:
        wd = word_diff(mo.group(1), mn.group(1))
        if wd is not None:
            return new[:mn.start(1)] + wd + new[mn.end(1):]
    return "> **Model D:** this figure or its caption is changed.\n\n" + new


def main() -> None:
    base, draft, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    csl = base.parent / "ieee.csl" if (base.parent / "ieee.csl").exists() else Path(sys.argv[4])
    mb, _ = to_markdown(base, csl)
    md, dates = to_markdown(draft, csl)
    tb, td = tidy(mb), tidy(md)
    cmap = cite_numbers(td["body"])
    for k in cite_numbers(tb["body"]):  # a reference the new version no longer cites keeps a number past the end
        cmap.setdefault(k, len(cmap) + 1)
    pandoc_first = re.search(r"\\\[(\d+)\\\]CITEKEYS([0-9a-f]+)ENDCITE", td["body"])
    assert pandoc_first and cmap[bytes.fromhex(pandoc_first.group(2)).decode().split(",")[0]] == int(pandoc_first.group(1))
    for t in (tb, td):
        for k in t:
            t[k] = recite(t[k], cmap)
    tb["body"] = renumber_refs(tb["body"], td["body"])
    (out / "_base.md").write_text(tb["body"], encoding="utf-8")
    (out / "_draft.md").write_text(td["body"], encoding="utf-8")
    body, stats = mark(tb["body"], td["body"])
    abstract = word_diff(tb["abstract"], td["abstract"]) or td["abstract"]
    subtitle = word_diff(tb["subtitle"], td["subtitle"]) or td["subtitle"]
    head = (Path(__file__).parent / "FULL_REPORT_HEADER.md").read_text(encoding="utf-8")
    text = head.replace("{{SUBTITLE}}", subtitle).replace("{{DATES}}", dates).replace("{{ABSTRACT}}", abstract) + "\n\n" + body + "\n"
    (out / "FULL_REPORT_WITH_MODEL_D.md").write_text(text, encoding="utf-8", newline="\n")
    figs = sorted(set(re.findall(r"figures/([A-Za-z0-9_]+\.png)", text)))
    print(json.dumps({"blocks": stats, "ins": text.count("<ins>"), "del": text.count("<del>"), "figures": figs,
                      "lines": text.count(chr(10))}, indent=1))


if __name__ == "__main__":
    main()

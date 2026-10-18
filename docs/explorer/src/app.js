/* compclass explorer — vanilla JS, no dependencies. Data comes from <script id="data"> (built by
   scripts/build_explorer.py from repository files); thumbnails from <script id="thumbs">. */
"use strict";
(function () {
// ---------------------------------------------------------------- error capture (visible in screenshots/QA)
const ERRORS = [];
function showErr(msg) {
  ERRORS.push(msg);
  const b = document.getElementById("errbar");
  if (b) { b.hidden = false; b.textContent = "JavaScript error(s):\n" + ERRORS.join("\n"); }
  try { document.title = "[JS ERROR] " + document.title.replace(/^\[JS ERROR\] /, ""); } catch (e) { /* ignore */ }
}
window.addEventListener("error", (e) => showErr((e.message || "error") + " @" + (e.lineno || "?") + ":" + (e.colno || "?")));
window.addEventListener("unhandledrejection", (e) => showErr("unhandled rejection: " + (e.reason && e.reason.message || e.reason)));
window.__explorerErrors = ERRORS;

const D = JSON.parse(document.getElementById("data").textContent);
const TH = JSON.parse(document.getElementById("thumbs").textContent);
const C = D.classes, K = C.length;

// ---------------------------------------------------------------- small utilities
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const f2 = (x) => (x == null ? "—" : (+x).toFixed(2));
const f3 = (x) => (x == null ? "—" : (Math.round(x * 1000 + 1e-9) / 1000).toFixed(3));
const f4 = (x) => (x == null ? "—" : (+x).toFixed(4));
const pct = (x, d = 1) => (x == null ? "—" : (100 * x).toFixed(d) + " %");
const pts = (x, d = 1) => (x == null ? "—" : (x > 0 ? "+" : x < 0 ? "−" : "±") + Math.abs(x).toFixed(d));
const num = (x) => (x == null ? "—" : Number(x).toLocaleString("en-US"));
const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
const gcls = (g) => "g-" + String(g).replace(/ /g, "-");
const scls = (g) => "s-" + String(g).replace(/ /g, "-");
const GROUPS = ["Classical", "From scratch", "ImageNet CNN", "ImageNet transformer", "Self-supervised", "Vision-language"];
let UID = 0;
const uid = (p) => (p || "u") + (++UID);
function wilson(k, n, z = 1.959964) {
  if (!n) return [0, 0];
  const p = k / n, d = 1 + z * z / n, c = (p + z * z / (2 * n)) / d;
  const h = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d;
  return [c - h, c + h];
}
function img(key, alt, cls) {
  const b = TH[key];
  if (!b) return `<span class="muted small">[no image]</span>`;
  return `<img src="data:image/jpeg;base64,${b}" alt="${esc(alt || "")}" loading="lazy" ${cls ? `class="${cls}"` : ""}>`;
}
function thumb(key, cap, tip, cls) {
  return `<figure class="th ${cls || ""}" ${tip ? `data-tip="${esc(tip)}"` : ""}>${img(key, cap)}<figcaption>${cap}</figcaption></figure>`;
}
function legendGroups(groups) {
  return `<div class="legend">${(groups || GROUPS).map((g) => `<span><svg width="12" height="12" style="display:inline;vertical-align:middle"><circle cx="6" cy="6" r="5" class="${gcls(g)}"/></svg> ${esc(g)}</span>`).join("")}</div>`;
}
function seg(id, opts, cur) {
  return `<div class="seg" role="group" id="${id}">${opts.map(([v, l]) => `<button type="button" data-v="${esc(v)}" aria-pressed="${v === cur}">${esc(l)}</button>`).join("")}</div>`;
}
function onSeg(root, id, fn) {
  const s = $("#" + id, root);
  if (!s) return;
  s.addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    $$("button", s).forEach((x) => x.setAttribute("aria-pressed", x === b));
    fn(b.dataset.v);
  });
}
function chips(id, opts, on) {
  return `<div class="chips" role="group" id="${id}">${opts.map(([v, l]) => `<button type="button" class="chip" data-v="${esc(v)}" aria-pressed="${on.includes(v)}">${l}</button>`).join("")}</div>`;
}
function onChips(root, id, fn) {
  const s = $("#" + id, root);
  s.addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    b.setAttribute("aria-pressed", b.getAttribute("aria-pressed") !== "true");
    fn($$("button", s).filter((x) => x.getAttribute("aria-pressed") === "true").map((x) => x.dataset.v));
  });
}
function srcNote(s) { return `<div class="tiny muted">Source: ${s.split(/,\s*/).map((x) => `<code>${esc(x)}</code>`).join(", ")}</div>`; }

// ---------------------------------------------------------------- tooltip (any element with data-tip; data-tipimg = thumb key)
const tip = document.getElementById("tip");
document.addEventListener("mouseover", (e) => {
  const t = e.target.closest("[data-tip]");
  if (!t) { tip.hidden = true; return; }
  const k = t.getAttribute("data-tipimg");
  tip.innerHTML = esc(t.getAttribute("data-tip")).replace(/\n/g, "<br>") + (k && TH[k] ? img(k) : "");
  tip.hidden = false;
});
document.addEventListener("mousemove", (e) => {
  if (tip.hidden) return;
  const w = tip.offsetWidth, h = tip.offsetHeight;
  let x = e.clientX + 14, y = e.clientY + 14;
  if (x + w > window.innerWidth - 6) x = e.clientX - w - 12;
  if (y + h > window.innerHeight - 6) y = e.clientY - h - 12;
  tip.style.left = Math.max(4, x) + "px"; tip.style.top = Math.max(4, y) + "px";
});
document.addEventListener("focusin", (e) => {
  const t = e.target.closest && e.target.closest("[data-tip]");
  if (!t) { tip.hidden = true; return; }
  const r = t.getBoundingClientRect();
  tip.textContent = t.getAttribute("data-tip"); tip.hidden = false;
  tip.style.left = Math.min(window.innerWidth - 300, r.left) + "px"; tip.style.top = (r.bottom + 6) + "px";
});

// ---------------------------------------------------------------- SVG chart helpers (string builders)
function lin(d0, d1, r0, r1) { const k = (r1 - r0) / ((d1 - d0) || 1); const f = (v) => r0 + (v - d0) * k; f.inv = (p) => d0 + (p - r0) / k; return f; }
function logS(d0, d1, r0, r1) { const a = Math.log10(d0), b = Math.log10(d1); const f = (v) => r0 + (Math.log10(v) - a) * (r1 - r0) / (b - a); return f; }
function ticks(a, b, n = 5) {
  const span = b - a, step0 = Math.pow(10, Math.floor(Math.log10(span / n))), err = span / n / step0;
  const step = step0 * (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1);
  const out = []; for (let v = Math.ceil(a / step) * step; v <= b + 1e-9; v += step) out.push(+v.toFixed(10)); return out;
}
function logTicks(a, b) { const out = []; for (let e = Math.floor(Math.log10(a)); e <= Math.ceil(Math.log10(b)); e++) [1, 2, 5].forEach((m) => { const v = m * Math.pow(10, e); if (v >= a * 0.999 && v <= b * 1.001) out.push(v); }); return out; }
function svgOpen(w, h, extra) { return `<svg viewBox="0 0 ${w} ${h}" width="${w}" role="img" ${extra || ""}>`; }
const tfmt = (v) => (Math.abs(v) >= 1000 ? (v / 1000) + "k" : Math.abs(v) < 0.01 && v !== 0 ? v.toExponential(0) : +v.toFixed(3) + "");
function xAxis(x, y0, y1, tv, fmt, label, w0, w1) {
  let s = `<g class="grid">${tv.map((v) => `<line x1="${x(v)}" x2="${x(v)}" y1="${y0}" y2="${y1}"/>`).join("")}</g>`;
  s += `<g>${tv.map((v) => `<text x="${x(v)}" y="${y1 + 14}" text-anchor="middle" class="sm mut">${(fmt || tfmt)(v)}</text>`).join("")}</g>`;
  if (label) s += `<text x="${(w0 + w1) / 2}" y="${y1 + 30}" text-anchor="middle" class="sm">${esc(label)}</text>`;
  return s;
}
function yAxis(y, x0, x1, tv, fmt, label, h0, h1) {
  let s = `<g class="grid">${tv.map((v) => `<line x1="${x0}" x2="${x1}" y1="${y(v)}" y2="${y(v)}"/>`).join("")}</g>`;
  s += `<g>${tv.map((v) => `<text x="${x0 - 5}" y="${y(v) + 4}" text-anchor="end" class="sm mut">${(fmt || tfmt)(v)}</text>`).join("")}</g>`;
  if (label) s += `<text transform="translate(${x0 - 40},${(h0 + h1) / 2}) rotate(-90)" text-anchor="middle" class="sm">${esc(label)}</text>`;
  return s;
}
/** Horizontal bar chart. items: {label, v, cls, tip, text, sub} */
function barsH(items, o = {}) {
  const W = o.w || 640, lw = o.lw || 190, bh = o.bh || 20, gap = o.gap || 6, rw = o.rw || 70;
  const H = items.length * (bh + gap) + 28;
  const vmax = o.max != null ? o.max : Math.max(...items.map((d) => d.v)) * 1.05;
  const x = lin(o.min || 0, vmax, lw, W - rw);
  let s = svgOpen(W, H, `aria-label="${esc(o.aria || "bar chart")}"`);
  s += xAxis(x, 0, H - 24, ticks(o.min || 0, vmax, o.nt || 5), o.fmt, null);
  items.forEach((d, i) => {
    const y = i * (bh + gap) + 2;
    s += `<text x="${lw - 6}" y="${y + bh / 2 + 4}" text-anchor="end" class="lbl">${esc(d.label)}</text>`;
    s += `<rect x="${lw}" y="${y}" width="${Math.max(0, x(d.v) - lw)}" height="${bh}" rx="3" class="${d.cls || "g-ImageNet-CNN"} hov" data-tip="${esc(d.tip || d.label + ": " + d.v)}"/>`;
    if (d.v2 != null) s += `<rect x="${lw}" y="${y}" width="${Math.max(0, x(d.v2) - lw)}" height="${bh}" rx="3" class="${d.cls2}" data-tip="${esc(d.tip2 || "")}"/>`;
    s += `<text x="${x(d.v) + 5}" y="${y + bh / 2 + 4}" class="sm">${esc(d.text != null ? d.text : d.v)}</text>`;
  });
  if (o.ref != null) s += `<line x1="${x(o.ref)}" x2="${x(o.ref)}" y1="0" y2="${H - 24}" stroke="var(--fg2)" stroke-dasharray="4 3"/><text x="${x(o.ref) + 3}" y="${H - 27}" class="sm mut">${esc(o.refl || "")}</text>`;
  return s + "</svg>";
}

// ---------------------------------------------------------------- tabs + routing
const TABS = [
  ["overview", "Overview", renderOverview], ["data", "Data audit", renderData], ["protocol", "Evaluation protocol", renderProtocol],
  ["bench", "Benchmark explorer", renderBench], ["ablations", "Ablations", renderAblations], ["final", "Final model", renderFinal],
  ["general", "Generalisation", renderGeneral], ["families", "Model families", renderFamilies], ["glossary", "Glossary", renderGlossary],
  ["reproduce", "Reproduce", renderReproduce],
];
const nav = document.getElementById("tabs"), main = document.getElementById("main");
const rendered = {};
nav.innerHTML = TABS.map(([id, t], i) => `<button type="button" role="tab" id="tab-${id}" aria-controls="panel-${id}" aria-selected="false" tabindex="${i ? -1 : 0}">${t}</button>`).join("");
main.innerHTML = TABS.map(([id]) => `<section role="tabpanel" id="panel-${id}" aria-labelledby="tab-${id}" hidden tabindex="-1"></section>`).join("");
function activate(id, sub, push) {
  if (!TABS.find((t) => t[0] === id)) id = "overview";
  TABS.forEach(([t]) => {
    const on = t === id, b = $("#tab-" + t), p = $("#panel-" + t);
    b.setAttribute("aria-selected", on); b.tabIndex = on ? 0 : -1; p.hidden = !on;
  });
  const panel = $("#panel-" + id);
  if (!rendered[id]) {
    try { TABS.find((t) => t[0] === id)[2](panel); rendered[id] = true; }
    catch (err) { showErr(`render ${id}: ${err.message}\n${(err.stack || "").split("\n").slice(0, 3).join("\n")}`); }
  }
  if (panel.__sub && sub != null) panel.__sub(sub, !!push);
  const h = "#" + id + (sub ? "/" + sub : "");
  if (push && location.hash !== h) { try { history.replaceState(null, "", h); } catch (e) { /* sandboxed */ } }
}
nav.addEventListener("click", (e) => { const b = e.target.closest("button[role=tab]"); if (b) { activate(b.id.slice(4), null, true); window.scrollTo(0, 0); } });
nav.addEventListener("keydown", (e) => {
  const ids = TABS.map((t) => t[0]), cur = ids.indexOf(document.activeElement.id.slice(4));
  if (cur < 0) return;
  let nx = null;
  if (e.key === "ArrowRight") nx = (cur + 1) % ids.length; else if (e.key === "ArrowLeft") nx = (cur - 1 + ids.length) % ids.length;
  else if (e.key === "Home") nx = 0; else if (e.key === "End") nx = ids.length - 1;
  if (nx != null) { e.preventDefault(); $("#tab-" + ids[nx]).focus(); activate(ids[nx], null, true); }
});
function go(id, sub) { activate(id, sub, true); window.scrollTo(0, 0); $("#tab-" + id).focus({ preventScroll: true }); }
document.addEventListener("click", (e) => { const a = e.target.closest("[data-go]"); if (a) { e.preventDefault(); const [t, s] = a.dataset.go.split("/"); go(t, s); } });

// theme
function setTheme(t) { document.documentElement.setAttribute("data-theme", t); try { localStorage.setItem("cc-theme", t); } catch (e) { /* sandbox */ } }
let theme0 = null;
try { theme0 = localStorage.getItem("cc-theme"); } catch (e) { /* sandbox */ }
const hashTheme = /theme=(dark|light)/.exec(location.hash);
if (hashTheme) theme0 = hashTheme[1];
else if (!theme0 && window.matchMedia && matchMedia("(prefers-color-scheme: dark)").matches) theme0 = "dark";
setTheme(theme0 || "light");
$("#themebtn").addEventListener("click", () => setTheme(document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark"));
$("#buildinfo").textContent = `data v2 ${D.meta.manifest}` + (D.meta.head ? ` · git ${D.meta.head} (${D.meta.branch})` : "");

// ==================================================================================================== 1. OVERVIEW
function renderOverview(el) {
  const t = D.final.test, w = D.final.web, k = D.kaggle, best = D.bench.find((b) => b.name === "probe_siglip_so400m_384");
  const ft = D.bench.filter((b) => b.schedule === "fixed" && b.regime === "fine-tune");
  const bestFT = ft.reduce((a, b) => (b.f1 > a.f1 ? b : a));
  const partial = D.bench.find((b) => b.name === "siglip_so400m_384_partial_v2");
  el.innerHTML = `
  <h2>What this project did</h2>
  <p class="lead">Goal: compare model families for recognising <b>20 kinds of electronic components</b> in internet photos, on an
  evaluation that cannot be fooled by duplicate images, and ship <b>one exportable model at about 80–85 % accuracy</b> that says
  <i>UNSURE</i> when it should. The data is a web-scraped Kaggle set; its audit turned out to be half of the story.</p>
  <div class="grid g3">
    <div class="card kpi"><div class="v">${pct(t.accuracy)}</div><div class="l">Sealed-test accuracy</div>
      <div class="s">n = ${t.n}, one look · 95 % CI ${pct(t.ci[0])}–${pct(t.ci[1])} · macro-F1 ${f3(t.macro_f1)} · SE ±${D.kn.test_accuracy_se_pts} pt</div></div>
    <div class="card kpi"><div class="v">${pct(w.accuracy)}</div><div class="l">Web-holdout accuracy (Wikimedia photos)</div>
      <div class="s">n = ${w.n}, collected after the model was frozen · 95 % CI ${pct(w.ci[0])}–${pct(w.ci[1])} · macro-F1 ${f3(w.macro_f1)}</div></div>
    <div class="card kpi"><div class="v">${pct(t.coverage)} <span style="font-size:1.1rem">@ ${pct(t.accepted_accuracy)}</span></div><div class="l">Answers / accuracy when answering</div>
      <div class="s">abstains below confidence ${D.fit.abstain_threshold} (sealed test) · try the slider in <a href="#final" data-go="final">Final model</a></div></div>
  </div>
  <div class="grid g2">
    <div class="card"><h3>The four research questions, answered</h3><ol>
      <li><b>Which family wins when evaluation is leakage-free?</b> Web-scale vision-language pretraining. A frozen
        ${esc(best.label)} probe reaches macro-F1 <b>${f3(best.f1)} ± ${f3(best.f1_sd)}</b> on grouped 5-fold CV; the best full fine-tune
        (${esc(bestFT.label)}) ${f3(bestFT.f1)}, the partial fine-tune of the same So400m backbone ${f3(partial.f1)} (folds 0–2); ImageNet fine-tunes ${f3(D.kn.imagenet_finetune_f1_range[0])}–${f3(D.kn.imagenet_finetune_f1_range[1])};
        scratch CNN ${f3(D.bench.find((b) => b.name === "scratch_cnn" && b.schedule === "fixed").f1)}; HOG + SVM ${f3(D.bench.find((b) => b.name === "hog_hsv_svm").f1)}.
        <a href="#bench" data-go="bench">Benchmark →</a></li>
      <li><b>Freeze or fine-tune at ~2.5k images?</b> It depends on the backbone: fine-tuning beats the probe for ImageNet, DINOv2 and
        CLIP-B (${pts(Math.min(...D.paired.filter((p) => p.mean_delta > 0).map((p) => p.mean_delta)))} to ${pts(Math.max(...D.paired.map((p) => p.mean_delta)))} pts, 5/5 folds) but <i>not</i> for SigLIP. <a href="#ablations" data-go="ablations">Ablations →</a></li>
      <li><b>What does accuracy cost?</b> Accuracy follows pretraining data, not compute: a ~23k-parameter logistic regression on frozen
        SigLIP So400m features beats a 76M-parameter partial fine-tune. The real cost is the frozen backbone's inference
        (${D.export.ort_cpu_ms_per_image} ms/image on CPU). <a href="#bench/scatter" data-go="bench/scatter">Cost scatter →</a></li>
      <li><b>How good is the final model on unseen data?</b> ${pct(t.accuracy)} on the sealed test, ${pct(w.accuracy)} on fresh web photos;
        on the original Kaggle files it has never seen, ${pct(k.by_split.test.accuracy)} (n = ${k.by_split.test.n}).
        <a href="#general" data-go="general">Generalisation →</a></li>
    </ol></div>
    <div class="card"><h3>Read this before quoting any number</h3><ul class="small">
      <li>Frozen-probe ≥ fine-tune holds <b>only for SigLIP</b>; ImageNet, DINOv2 and CLIP-B fine-tunes beat their own probes.</li>
      <li>Zero-shot fusion is <b>null in-distribution</b> (test ${f3(t.accuracy)} fused vs ${f3(t.probe_only_accuracy)} probe-only) but
        <b>+${(100 * (w.accuracy - w.probe_only_accuracy)).toFixed(1)} pts</b> on the web holdout.</li>
      <li>The LR-schedule fix is <b>confounded</b> with BatchNorm freezing (shipped in one commit).</li>
      <li>Out-of-scope flags are an <b>automated CLIP screen</b>; the 69 label issues are model-flagged. Neither was human-verified.</li>
      <li>The web holdout was curated by <b>one automated reviewer</b> (clear single-component photos), so it is easier than a random web sample.</li>
      <li>Test SE is about <b>±${D.kn.test_accuracy_se_pts} pt</b>; test differences under ~3.5 pts are not resolvable. One seed per configuration; one source dataset.</li>
      <li>OpenCV DNN <b>cannot load</b> the ONNX graph (segfault), so inference uses onnxruntime.</li>
      <li>The original-Kaggle headline (${pct(k.all.accuracy)}) includes the files used to fit the head; only the ${k.by_split.test.n} never-seen files are honest (${pct(k.by_split.test.accuracy)}).</li>
    </ul></div>
  </div>
  <div class="card"><h3>Project timeline</h3><p class="small muted">Click a phase (or use Tab + Enter) to see what was done, why, the result, the files and the commits
    (from <code>git log</code>, ${D.meta.n_commits} commits on <code>${esc(D.meta.branch || "?")}</code>).</p>
    <div class="tl" role="group" aria-label="Project phases">${D.timeline.map((p, i) => `<button type="button" data-i="${i}" aria-pressed="${i === 6}"><div class="d">${esc(p.date)}</div><div class="n">${esc(p.title)}</div></button>`).join("")}</div>
    <div id="tldetail"></div></div>
  <div class="card"><h3>How to use this explorer</h3><p class="small">Tabs follow the order of the work: the <b>data audit</b> explains why the
  evaluation had to be rebuilt; the <b>protocol</b> shows how leakage was prevented and how much it inflated the old numbers; the
  <b>benchmark explorer</b> and <b>ablations</b> hold every measured result; <b>final model</b> and <b>generalisation</b> cover the
  deployed classifier; <b>model families</b> is a self-contained course on each architecture (diagram, tensor shapes, pretraining,
  hyperparameters with config keys, results here). Every number is read from a repository file at build time
  (<code>scripts/build_explorer.py</code>); the build fails if a headline number disagrees with its source.</p></div>`;
  const det = $("#tldetail", el);
  function show(i) {
    const p = D.timeline[i];
    $$(".tl button", el).forEach((b) => b.setAttribute("aria-pressed", +b.dataset.i === i));
    det.innerHTML = `<div class="grid g21"><div><h4>${esc(p.title)} <span class="badge">${esc(p.date)}</span></h4>
      <dl class="kv"><dt>What</dt><dd>${p.what}</dd><dt>Why</dt><dd>${p.why}</dd><dt>Result</dt><dd>${p.result}</dd>
      <dt>Files</dt><dd>${p.files.map((f) => `<code>${esc(f)}</code>`).join(" ")}</dd></dl></div>
      <div><h4>Commits</h4><ul class="commits">${p.commits.map((c) => `<li><code>${esc(c.h)}</code> <span class="muted">${esc(c.d)}</span> ${esc(c.s)}</li>`).join("")}</ul></div></div>`;
  }
  $(".tl", el).addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) show(+b.dataset.i); });
  show(6);
}

// ==================================================================================================== 2. DATA AUDIT
function renderData(el) {
  const A = D.audit, c = A.counts;
  const funnel = [
    { label: "Raw image files", v: c.raw_files, text: num(c.raw_files), tip: "every file in the Kaggle download", cls: "g-Classical" },
    { label: "Unique (SHA-256)", v: c.unique_sha256, text: `${num(c.unique_sha256)}  (−${num(c.exact_dup_redundant_files)} copies in ${num(c.exact_dup_groups)} groups)`, tip: "exact duplicates collapsed", cls: "g-ImageNet-CNN" },
    { label: "Rows kept", v: c.rows, text: `${num(c.rows)}  (−${c.exact_dup_conflicting_label_groups} conflicting-label groups)`, tip: "conflicting-label exact duplicates excluded", cls: "g-ImageNet-transformer" },
    { label: "Duplicate clusters", v: c.clusters, text: `${num(c.clusters)}  (${c.rows_in_near_dup_clusters} rows in ${c.near_dup_clusters} near-dup clusters)`, tip: "the grouping unit for every split", cls: "g-Vision-language" },
  ];
  const flagged = [
    { label: "Flagged out-of-scope", v: c.ood_suspect, text: `${num(c.ood_suspect)} (${pct(c.ood_suspect / c.rows)}) — ${A.ood_dev} dev / ${A.ood_test} test`, cls: "g-Self-supervised", tip: "CLIP distractor screen, threshold " + A.ood_rule.threshold },
    { label: "In near-dup clusters", v: c.rows_in_near_dup_clusters, text: num(c.rows_in_near_dup_clusters), cls: "g-From-scratch" },
    { label: "Copies across legacy splits", v: A.n_legacy_multi, text: num(A.n_legacy_multi) + " rows", cls: "g-From-scratch", tip: "rows whose duplicate files sat in different legacy splits (flag legacy_multi_split)" },
    { label: "Alpha-composited", v: c.flags.alpha_composited, text: num(c.flags.alpha_composited), cls: "g-Classical" },
  ];
  el.innerHTML = `
  <h2>Data audit: what is really in the Kaggle download?</h2>
  <p class="lead">Before comparing models, every file was decoded, hashed, linked to its near-copies and screened for content that is
  not an electronic component (<code>tools/build_dataset_v2.py</code> → <code>data/v2/manifest.json</code>, fingerprint <code>${esc(D.meta.manifest)}</code>).</p>
  <div class="grid g2">
    <div class="card"><h3>Cleaning funnel</h3>${barsH(funnel, { w: 620, lw: 130, rw: 250, max: c.raw_files * 1.0, nt: 4, aria: "cleaning funnel" })}
      <h4 style="margin-top:12px">Flags on retained rows (kept, not removed)</h4>${barsH(flagged, { w: 620, lw: 170, rw: 190, max: c.ood_suspect, nt: 4 })}
      ${srcNote("data/v2/leakage_report.json")}</div>
    <div class="card"><h3>In plain words</h3>
      <p>${D.explain.exact}</p><p>${D.explain.near}</p><p>${D.explain.conflict}</p>
      <p class="small muted">Near-duplicate rule: pHash Hamming ≤ ${A.near_rule.phash_radius} with CLIP cosine ≥ ${A.near_rule.cos_with_phash},
      or cosine ≥ ${A.near_rule.cos_only} alone. Cluster sizes: ${Object.entries(A.near_hist).map(([s, n]) => `${n}×${s}`).join(", ")}.
      Unique formats: ${Object.entries(A.formats).map(([f, n]) => `${f} ${num(n)}`).join(", ")}.</p></div>
  </div>
  <div class="grid g21">
    <div class="card"><h3>Per class: rows and out-of-scope share</h3>
      <div class="ctl"><label>Sort</label>${seg("pcsort", [["share", "by OOD share"], ["n", "by rows"], ["name", "by name"]], "share")}</div>
      <div id="pcchart"></div>
      <p class="small">Contamination concentrates in classes whose name is a common English word: <b>switch</b> (Nintendo Switch),
      <b>transformer</b> (toy robots), <b>relay</b>, <b>fuse</b> (brand logos). ${srcNote("data/v2/manifest.json (flags)")}</p></div>
    <div class="card"><h3>Top distractor concept of flagged rows</h3>${barsH(Object.entries(A.distractors).map(([k, v]) => ({ label: k.replace(/_/g, " "), v, text: v, cls: "g-Self-supervised" })), { w: 420, lw: 130, rw: 30, nt: 3 })}
      <p class="small">${D.explain.ood}</p>${srcNote("data/v2/review/ood_suspects.csv")}</div>
  </div>
  <div class="card"><h3>Look at the images</h3>
    <div class="ctl">${seg("galsel", [["ood", "Out-of-scope"], ["exact", "Copies across legacy splits"], ["near", "Near-duplicates"], ["big", "Largest cluster"], ["conflict", "Conflicting labels"], ["issues", `Label issues (${A.n_issues})`], ["classes", "Typical examples"]], "ood")}</div>
    <div id="galbody"></div>
    <p class="tiny muted">Thumbnails (112 px) are generated at build time from <code>data/v2/images</code> and <code>data/kaggle_eval_extra/images</code>.</p></div>`;
  // per-class chart
  function pc(mode) {
    const rows = A.per_class.slice();
    if (mode === "share") rows.sort((a, b) => b.ood / b.n - a.ood / a.n); else if (mode === "n") rows.sort((a, b) => b.n - a.n); else rows.sort((a, b) => a.cls.localeCompare(b.cls));
    const W = 720, lw = 150, bh = 15, gap = 4, H = rows.length * (bh + gap) + 30, maxn = Math.max(...rows.map((r) => r.n));
    const x = lin(0, maxn * 1.02, lw, W - 110);
    let s = svgOpen(W, H, 'aria-label="rows per class"') + xAxis(x, 0, H - 26, ticks(0, maxn, 5), null, null);
    rows.forEach((r, i) => {
      const y = i * (bh + gap) + 2, sh = r.ood / r.n;
      s += `<text x="${lw - 6}" y="${y + 12}" text-anchor="end" class="lbl">${esc(r.cls)}</text>`;
      s += `<rect x="${lw}" y="${y}" width="${x(r.n - r.ood) - lw}" height="${bh}" class="g-ImageNet-CNN" data-tip="${esc(`${r.cls}: ${r.n} rows, ${r.n - r.ood} in scope, ${r.ood} flagged (${pct(sh)}); ${r.test} in sealed test`)}"/>`;
      s += `<rect x="${x(r.n - r.ood)}" y="${y}" width="${x(r.n) - x(r.n - r.ood)}" height="${bh}" class="g-Self-supervised" data-tip="${esc(`${r.cls}: ${r.ood} flagged out-of-scope (${pct(sh)})`)}"/>`;
      s += `<text x="${x(r.n) + 5}" y="${y + 12}" class="sm">${r.n} · ${pct(sh, 0)}</text>`;
    });
    $("#pcchart", el).innerHTML = s + "</svg>" + `<div class="legend"><span><span class="dot" style="background:var(--g-incnn)"></span>in scope</span><span><span class="dot" style="background:var(--g-ssl)"></span>flagged out-of-scope (overall ${pct(c.ood_suspect / c.rows)})</span></div>`;
  }
  onSeg(el, "pcsort", pc); pc("share");
  // galleries
  const G = D.gal, body = $("#galbody", el);
  function gal(which) {
    let h = "";
    if (which === "ood") {
      const ds = Object.keys(A.distractors);
      h = `<p class="small">${D.explain.ood}</p><div class="ctl">${chips("oodf", ds.map((d) => [d, d.replace(/_/g, " ")]), ds)}</div><div class="gallery" id="oodg"></div>`;
      body.innerHTML = h;
      const draw = (on) => { $("#oodg", body).innerHTML = G.ood.filter((g) => on.includes(g.distractor)).map((g) => thumb(g.id, `<b>${esc(g.label)}</b><br>${esc(g.distractor.replace(/_/g, " "))} p=${g.p}`, `${g.id} (${g.split})\nlabel: ${g.label}\ntop distractor: ${g.distractor} p=${g.p}\nOOD score ${g.score}`)).join(""); };
      onChips(body, "oodf", draw); draw(ds); return;
    }
    if (which === "exact") h = `<p class="small">${D.explain.exact} ${A.n_legacy_multi} rows carry the <code>legacy_multi_split</code> flag.</p><div class="gallery">${G.exact.map((g) => thumb(g.id, `<b>${esc(g.label)}</b><br>${g.copies} copies: ${esc(Array.from(new Set(g.legacy)).join(" + "))}`, `${g.copies} identical files; legacy splits: ${g.legacy.join(", ")}\n` + g.files.join("\n"), "bad")).join("")}</div>`;
    if (which === "near") h = `<p class="small">${D.explain.near}</p><div class="gallery">${G.near.map((p) => `<div class="pair">${thumb(p.a, esc(p.la))}${thumb(p.b, `${esc(p.lb)}<br><span class="muted">pHash ${p.phash} · cos ${p.cos} · ${esc(p.rule)}</span>`)}</div>`).join("")}</div>`;
    if (which === "big") h = `<p class="small">The largest near-duplicate cluster has <b>${G.near_big.size}</b> rows (label: ${esc(G.near_big.labels)}). A file-level split scatters such a cluster over train and test; the grouped split keeps it on one side. First ${G.near_big.ids.length} shown.</p><div class="gallery">${G.near_big.ids.map((i) => thumb(i, i)).join("")}</div>`;
    if (which === "conflict") h = `<p class="small">${D.explain.conflict}</p><div class="gallery">${G.conflict.map((g) => thumb(g.key, `<b>${esc(g.labels.join(" vs "))}</b><br>${g.kind === "exact" ? "exact copy — excluded" : `near-dup cos ${g.cos} — kept, flagged`}`, `${g.group}: ${g.decision}`, g.kind === "exact" ? "bad" : "uns")).join("")}</div>`;
    if (which === "issues") {
      const pats = A.issue_patterns.map(([p]) => p);
      body.innerHTML = `<p class="small">${D.explain.issues} ${A.n_issues_ood} of the ${A.n_issues} are also out-of-scope-flagged. Most common patterns: ${A.issue_patterns.slice(0, 5).map(([p, n]) => `${esc(p)} (${n})`).join(", ")}.</p>
        <div class="ctl"><label>Show</label>${seg("isf", [["all", "all"], ["ood", "OOD-flagged"], ["in", "not OOD-flagged"]], "all")} <label>given → predicted</label><select id="isp"><option value="">any pattern</option>${pats.map((p) => `<option>${esc(p)}</option>`).join("")}</select></div><div class="gallery" id="isg"></div>`;
      let mode = "all";
      const draw = () => { const pp = $("#isp", body).value; $("#isg", body).innerHTML = G.issues.filter((g) => (mode === "all" || (mode === "ood") === g.ood) && (!pp || `${g.label} -> ${g.pred}` === pp)).map((g) => thumb(g.id, `given <b>${esc(g.label)}</b><br>→ ${esc(g.pred)}`, `${g.id}\np(given label) = ${g.self}\np(${g.pred}) = ${g.alt}${g.ood ? "\nalso OOD-flagged" : ""}`, "uns")).join(""); };
      onSeg(body, "isf", (v) => { mode = v; draw(); }); $("#isp", body).addEventListener("change", draw); draw(); return;
    }
    if (which === "classes") {
      body.innerHTML = `<p class="small">${D.explain.classes}</p><div class="ctl"><label>Class</label><select id="clsel">${C.map((c) => `<option>${esc(c)}</option>`).join("")}</select></div><div class="gallery" id="clg"></div>`;
      const draw = () => { const c = $("#clsel", body).value; $("#clg", body).innerHTML = (G.classes[c] || []).map((i) => thumb(i, esc(c))).join(""); };
      $("#clsel", body).addEventListener("change", draw); draw(); return;
    }
    body.innerHTML = h;
  }
  onSeg(el, "galsel", gal); gal("ood");
  el.__sub = (s) => { const b = $(`#galsel button[data-v="${s}"]`, el); if (b) b.click(); };
}

// ==================================================================================================== 3. PROTOCOL
function mulberry(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
function renderProtocol(el) {
  const A = D.audit, L = D.leakage, rp = D.kaggle.replay, fs = A.fold_sizes;
  el.innerHTML = `
  <h2>Evaluation protocol: grouped CV, a sealed test and a second, out-of-source holdout</h2>
  <div class="card"><h3>The splits (click a fold)</h3><div id="splitdia"></div><div id="splittext" class="small"></div>
    ${srcNote("data/v2/manifest.json, data/v2/leakage_report.json, data/web_holdout_v1/manifest.json")}</div>
  <div class="grid g2">
    <div class="card"><h3>Why grouping by cluster matters — a toy simulation</h3>
      <p class="small">Each dot is one <b>file</b>; files of the same colour are copies of one picture (a cluster). The split puts ~30 % of
      the data on the evaluation side. <b>Legacy</b> mode splits files independently; <b>grouped</b> mode splits whole clusters.
      Red lines connect evaluation files that have a copy in training — the model can answer them by memory.</p>
      <div class="ctl">${seg("leakmode", [["legacy", "Legacy: split files"], ["grouped", "v2: split clusters"]], "legacy")}<button class="btn" id="reshuf" type="button">Re-shuffle</button></div>
      <div id="toy"></div><div id="toytext" class="small"></div></div>
    <div class="card"><h3>Measured: how much the legacy split inflated accuracy</h3>
      <p class="small">${L.leaked} of the ${L.eval} legacy validation + test images (${pct(L.leaked / L.eval)}) duplicate a legacy training image.
      Replaying three linear probes under both protocols:</p><div id="leakchart"></div>
      <p class="small">On the duplicated rows the probes are nearly perfect, and <b>weaker models gain most</b> because memorised copies
      substitute for features they lack — leakage also <i>compresses</i> the gaps between families.</p>${srcNote("results/leakage_simulation.csv")}</div>
  </div>
  <div class="grid g2">
    <div class="card"><h3>Same files, same protocol: the old notebooks vs the final model</h3>
      <p class="small">The final model's backbone + fused head with the probe <b>re-fitted only on legacy-train rows</b>, scored on the
      legacy-test files — the old notebooks' exact protocol, leakage included:</p><div id="replaychart"></div>
      <div class="callout small">On an identical protocol the final model is <b>${pts(100 * (rp.all.accuracy - rp.old_best_for_reference.accuracy))} pts</b>
      above the old best (${esc(rp.old_best_for_reference.model)}, ${pct(rp.old_best_for_reference.accuracy)}). Even its score on files with no copy
      in training (${pct(rp.clean_not_in_train.accuracy)}, n = ${rp.clean_not_in_train.n}) is above the old leaked number.
      ${rp.excluded_conflict_files_not_scored} conflicting-label files were not scored.</div>${srcNote("results/kaggle_original/summary.json, legacy/docs/EXPERIMENT_LOG.md")}</div>
    <div class="card"><h3>Metrics and how precise they are</h3><ul class="small">
      <li><b>Macro-F1</b> (primary): mean of the 20 per-class F1 scores, so small classes count equally. Accuracy is secondary.</li>
      <li>CV numbers are <b>mean ± SD over 5 folds</b> of ~${Math.round(mean(fs))} images; fold SDs of 1–3.5 pts mix data and optimisation noise (one seed).</li>
      <li><b>In-scope</b> scores drop the automatically flagged out-of-scope rows (pooled over folds).</li>
      <li>Sealed test: n = ${D.final.test.n}, binomial SE ±${D.kn.test_accuracy_se_pts} pts; differences under ~3.5 pts between two models are not resolvable.</li>
      <li>Selection used dev folds only; the test split could be read only via <code>get_test(..., confirm=True)</code> and was read once.
        Fine-tunes early-stop on the fold they are scored on — a slight upward bias for fine-tunes.</li>
      <li>Largest class-distribution distance (TVD) across the outer grouped split: ${f3(A.outer_tvd_max)}.</li></ul></div>
  </div>`;
  // split diagram
  const W = 1000, H = 190, x0 = 20, x1 = W - 20, tot = A.counts.rows;
  const x = lin(0, tot, x0, x1);
  let sel = null;
  function drawSplit() {
    let s = svgOpen(W, H, 'class="dia" aria-label="split diagram"');
    s += `<text x="${x0}" y="16" class="t">data v2: ${num(tot)} rows in ${num(A.counts.clusters)} clusters</text>`;
    let cx = 0;
    const tw = A.counts.test_rows;
    s += `<rect class="box r" x="${x(0)}" y="26" width="${x(tw) - x0}" height="54" rx="6"/><text x="${(x(0) + x(tw)) / 2}" y="50" text-anchor="middle" class="t">sealed test</text><text x="${(x(0) + x(tw)) / 2}" y="68" text-anchor="middle" class="sm">${tw} rows · read once</text>`;
    cx = tw;
    fs.forEach((n, k) => {
      const cls = sel == null ? "a" : sel === k ? "o" : "g";
      s += `<g class="clk" data-fold="${k}" tabindex="0" role="button" aria-label="fold ${k}"><rect class="box ${cls}" x="${x(cx) + 2}" y="26" width="${x(cx + n) - x(cx) - 4}" height="54" rx="6"/><text x="${(x(cx) + x(cx + n)) / 2}" y="50" text-anchor="middle" class="t">fold ${k}</text><text x="${(x(cx) + x(cx + n)) / 2}" y="68" text-anchor="middle" class="sm">${n} rows${sel == null ? "" : sel === k ? " · validate" : " · train"}</text></g>`;
      cx += n;
    });
    s += `<path class="ln" d="M${x(tw)} 90 v8 H${x1} v-8"/><text x="${(x(tw) + x1) / 2}" y="114" text-anchor="middle" class="sm">dev = ${num(A.counts.dev_rows)} rows · 5 cluster-grouped folds · all training, tuning, calibration and model selection</text>`;
    s += `<rect class="box p" x="${x0}" y="130" width="${(x1 - x0) * 0.32}" height="46" rx="6"/><text x="${x0 + (x1 - x0) * 0.16}" y="150" text-anchor="middle" class="t">web holdout v1</text><text x="${x0 + (x1 - x0) * 0.16}" y="167" text-anchor="middle" class="sm">${D.web_meta.n} Wikimedia photos · built after freezing · read once</text>`;
    s += `<rect class="box m" x="${x0 + (x1 - x0) * 0.36}" y="130" width="${(x1 - x0) * 0.64}" height="46" rx="6"/><text x="${x0 + (x1 - x0) * 0.68}" y="150" text-anchor="middle" class="t">original Kaggle files (${num(D.kaggle.n_files)})</text><text x="${x0 + (x1 - x0) * 0.68}" y="167" text-anchor="middle" class="sm">post-hoc check · map to v2 rows → seen (dev) / never seen (test) / excluded</text>`;
    $("#splitdia", el).innerHTML = s + "</svg>";
    $("#splittext", el).innerHTML = sel == null ? "Each fold is validated once by a model trained on the other four; reported CV numbers are the mean ± SD of the five validation scores. Probes choose C by an inner grouped CV inside the training folds."
      : `Fold ${sel} as validation: train on ${num(A.counts.dev_rows - fs[sel])} rows of folds ${[0, 1, 2, 3, 4].filter((k) => k !== sel).join(", ")}, score on ${fs[sel]} rows. No cluster has members in both. The sealed test is never touched.`;
  }
  $("#splitdia", el).addEventListener("click", (e) => { const g = e.target.closest("[data-fold]"); if (g) { sel = sel === +g.dataset.fold ? null : +g.dataset.fold; drawSplit(); } });
  $("#splitdia", el).addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { const g = e.target.closest("[data-fold]"); if (g) { e.preventDefault(); sel = +g.dataset.fold; drawSplit(); $(`[data-fold="${sel}"]`, el).focus(); } } });
  drawSplit();
  // toy leakage simulation
  let seed = 7, mode = "legacy";
  function toy() {
    const rnd = mulberry(seed), clusters = [];
    for (let i = 0; i < 34; i++) { const r = rnd(); clusters.push(r < 0.62 ? 1 : r < 0.84 ? 2 : r < 0.95 ? 3 : 4); }
    const files = [];
    clusters.forEach((n, ci) => { const cs = rnd() < 0.3; for (let j = 0; j < n; j++) files.push({ c: ci, ev: mode === "grouped" ? cs : rnd() < 0.3 }); });
    const W = 560, colX = [30, 330], H = 60 + 21 * Math.ceil(Math.max(files.filter((f) => !f.ev).length, files.filter((f) => f.ev).length) / 10);
    const pos = {}, cnt = [0, 0];
    files.forEach((f, i) => { const side = f.ev ? 1 : 0, k = cnt[side]++; pos[i] = [colX[side] + (k % 10) * 21 + 10, 44 + Math.floor(k / 10) * 21]; });
    let s = svgOpen(W, H, 'class="dia" aria-label="toy split"');
    s += `<text x="${colX[0]}" y="24" class="t">train</text><text x="${colX[1]}" y="24" class="t">evaluation</text>`;
    let leaked = 0;
    const evalN = files.filter((f) => f.ev).length;
    files.forEach((f, i) => {
      if (!f.ev) return;
      const tr = files.findIndex((g, j) => g.c === f.c && !g.ev);
      if (tr >= 0) { leaked++; s += `<line class="ln r" x1="${pos[i][0]}" y1="${pos[i][1]}" x2="${pos[tr][0]}" y2="${pos[tr][1]}" opacity="0.7"/>`; }
    });
    files.forEach((f, i) => { const hue = (f.c * 47) % 360; s += `<circle cx="${pos[i][0]}" cy="${pos[i][1]}" r="7.5" fill="hsl(${hue} 65% 55%)" stroke="var(--card)" data-tip="cluster ${f.c} (${clusters[f.c]} file${clusters[f.c] > 1 ? "s" : ""})"/>`; });
    $("#toy", el).innerHTML = s + "</svg>";
    $("#toytext", el).innerHTML = `<span class="stat"><b class="${leaked ? "bad" : "good"}">${leaked}</b> of ${evalN} evaluation files have a copy in training</span>` +
      (mode === "legacy" ? "A memorising model gets these right for free." : "Grouped splitting makes this zero by construction.");
  }
  onSeg(el, "leakmode", (v) => { mode = v; toy(); }); $("#reshuf", el).addEventListener("click", () => { seed++; toy(); }); toy();
  // leakage chart: grouped vs legacy per model
  const rows = L.rows.slice().sort((a, b) => a.grouped_cv_acc - b.grouped_cv_acc);
  const pretty = { resnet50: "ResNet-50 probe", convnext_tiny: "ConvNeXt-T probe", siglip_so400m_384: "SigLIP So400m probe" };
  function dumb(rs, W) {
    const lw = 150, H = rs.length * 54 + 34, xx = lin(0.4, 1.0, lw, W - 20);
    let s = svgOpen(W, H, 'aria-label="leakage dumbbell"') + xAxis(xx, 0, H - 26, ticks(0.4, 1.0, 6), (v) => Math.round(v * 100) + "%", null);
    rs.forEach((r, i) => {
      const y = i * 54 + 16;
      s += `<text x="${lw - 6}" y="${y + 4}" text-anchor="end" class="lbl b">${esc(pretty[r.model] || r.model)}</text>`;
      s += `<line x1="${xx(r.grouped_cv_acc)}" x2="${xx(r.legacy_style_acc)}" y1="${y}" y2="${y}" stroke="var(--bad)" stroke-width="3"/>`;
      s += `<circle cx="${xx(r.grouped_cv_acc)}" cy="${y}" r="6" fill="var(--accent)" data-tip="grouped CV accuracy ${f4(r.grouped_cv_acc)}"/>`;
      s += `<circle cx="${xx(r.legacy_style_acc)}" cy="${y}" r="6" fill="var(--card)" stroke="var(--bad)" stroke-width="2" data-tip="legacy-style accuracy ${f4(r.legacy_style_acc)}"/>`;
      s += `<text x="${(xx(r.grouped_cv_acc) + xx(r.legacy_style_acc)) / 2}" y="${y - 9}" text-anchor="middle" class="sm bad">+${r.inflation_pts.toFixed(1)} pts</text>`;
      s += `<text x="${lw - 6}" y="${y + 20}" text-anchor="end" class="sm mut">dup. rows ${pct(r.acc_on_leaked)} · others ${pct(r.acc_on_clean)}</text>`;
      s += `<rect x="${xx(r.acc_on_clean)}" y="${y + 12}" width="3" height="10" fill="var(--fg2)" data-tip="accuracy on the ${L.eval - L.leaked} non-duplicated legacy rows ${f4(r.acc_on_clean)}"/><rect x="${xx(r.acc_on_leaked)}" y="${y + 12}" width="3" height="10" fill="var(--bad)" data-tip="accuracy on the ${L.leaked} duplicated rows ${f4(r.acc_on_leaked)}"/>`;
    });
    return s + "</svg>";
  }
  $("#leakchart", el).innerHTML = dumb(rows, 600) + `<div class="legend"><span><span class="dot" style="background:var(--accent)"></span>grouped CV</span><span><span class="dot" style="border:2px solid var(--bad)"></span>legacy-style split</span><span>ticks below: accuracy on duplicated (red) vs other legacy rows</span></div>`;
  $("#replaychart", el).innerHTML = barsH([
    { label: "Old notebooks' best", v: rp.old_best_for_reference.accuracy, text: `${pct(rp.old_best_for_reference.accuracy)} (${rp.old_best_for_reference.model})`, cls: "g-Classical" },
    { label: "Final model, replay", v: rp.all.accuracy, text: `${pct(rp.all.accuracy)} (n = ${rp.all.n})`, cls: "g-Vision-language" },
    { label: "  … leaked copies", v: rp.leaked_copy_of_train.accuracy, text: `${pct(rp.leaked_copy_of_train.accuracy)} (n = ${rp.leaked_copy_of_train.n})`, cls: "g-Self-supervised" },
    { label: "  … clean files", v: rp.clean_not_in_train.accuracy, text: `${pct(rp.clean_not_in_train.accuracy)} (n = ${rp.clean_not_in_train.n})`, cls: "g-ImageNet-CNN" },
  ], { w: 600, lw: 150, rw: 170, max: 1, fmt: (v) => Math.round(v * 100) + "%" });
}
// ==================================================================================================== 4. BENCHMARK
const FAM = Object.fromEntries(D.families.map((f) => [f.id, f]));
const METRICS = { f1: "macro-F1", acc: "accuracy", inscope_acc: "in-scope accuracy", inscope_f1: "in-scope macro-F1" };
const REGS = ["linear probe", "zero-shot", "fine-tune", "partial fine-tune", "from scratch", "classical"];
function marker(reg, cx, cy, r, cls, tipv) {
  const t = tipv ? `data-tip="${esc(tipv)}"` : "";
  if (reg === "linear probe") return `<circle cx="${cx}" cy="${cy}" r="${r}" class="${cls}" ${t}/>`;
  if (reg === "zero-shot") return `<path d="M${cx} ${cy - r * 1.25}L${cx + r * 1.25} ${cy}L${cx} ${cy + r * 1.25}L${cx - r * 1.25} ${cy}Z" class="${cls}" ${t}/>`;
  if (reg === "fine-tune" || reg === "partial fine-tune") return `<rect x="${cx - r}" y="${cy - r}" width="${2 * r}" height="${2 * r}" class="${cls}" ${reg === "partial fine-tune" ? 'fill-opacity="0.35" stroke-width="2" style="stroke:currentColor"' : ""} ${t}/>`;
  return `<path d="M${cx} ${cy - r * 1.2}L${cx + r * 1.1} ${cy + r * 0.9}L${cx - r * 1.1} ${cy + r * 0.9}Z" class="${cls}" ${t}/>`;
}
const regLegend = `<div class="legend">${REGS.map((r) => `<span><svg width="14" height="14" style="display:inline;vertical-align:middle">${marker(r, 7, 7, 5, "g-Classical")}</svg> ${r}</span>`).join("")}</div>`;
function benchTip(b) {
  return `${b.label} — ${b.regime}${b.schedule !== "-" ? " (" + b.schedule + " schedule)" : ""}\nmacro-F1 ${f4(b.f1)} ± ${f4(b.f1_sd)} · acc ${f4(b.acc)} ± ${f4(b.acc_sd)}\nin-scope acc ${f4(b.inscope_acc)} · ${b.folds} folds` +
    (b.trainable_M != null ? `\ntrainable ${b.trainable_M < 1 ? (b.trainable_M * 1e3).toFixed(1) + "k" : b.trainable_M.toFixed(1) + "M"} params` : "") +
    (b.train_min != null ? ` · ${b.train_min} min/fold · ${b.vram_gb} GB` : "");
}
function renderBench(el) {
  const st = { fams: D.families.map((f) => f.id), regs: REGS.slice(), sched: "main", metric: "f1", sort: "f1", dir: -1, xax: "trainable_M", logx: true };
  el.innerHTML = `
  <h2>Benchmark explorer — every method, grouped 5-fold CV on ${num(D.audit.counts.dev_rows)} dev images</h2>
  <div class="card"><div class="ctl"><label>Family</label>${chips("bfam", D.families.map((f) => [f.id, f.short]), st.fams)}</div>
    <div class="ctl"><label>Regime</label>${chips("breg", REGS.map((r) => [r, r]), st.regs)}
      <label>Fine-tune schedule</label>${seg("bsched", [["main", "fixed (main table)"], ["old", "old (bug) runs"], ["all", "all"]], "main")}
      <label>Metric</label><select id="bmet">${Object.entries(METRICS).map(([k, v]) => `<option value="${k}">${v}</option>`).join("")}</select></div>
    ${legendGroups()}${regLegend}<div id="bdots"></div>
    <p class="small muted">Bar = mean ± 1 SD over folds; small dots = individual folds (macro-F1/accuracy only). Hover anything for numbers.
    The SigLIP So400m partial fine-tunes ran on folds 0–2 only. ${srcNote("results/benchmark.csv, results/phase1/raw/runs.jsonl, results/phase1b/runs.jsonl, results/phase2/runs_fixed.jsonl, results/phase2/raw_oldschedule/runs_{A,B}.jsonl")}</p></div>
  <div class="card" id="scatter"><h3>Accuracy vs cost</h3>
    <div class="ctl"><label>x axis</label><select id="bx"><option value="trainable_M">trainable parameters (M) — probes = 20 × (D+1)</option><option value="train_min">fine-tune minutes per fold (RTX 4060 Ti)</option><option value="params_M">total parameters (M), fine-tunes</option><option value="vram_gb">peak training VRAM (GB)</option></select>
    <label><input type="checkbox" id="blog" checked> log x</label><label><input type="checkbox" id="bpar" checked> Pareto front</label></div>
    <div id="bscat"></div><p class="small muted">Zero-shot and classical rows have no trainable parameters / training cost and are omitted. Pareto front = no other shown method is both cheaper and better.</p></div>
  <div class="grid g2">
    <div class="card"><h3>Probe vs fine-tune, same backbone, same folds</h3><div id="bpair"></div>
      <p class="small">Each dot is one fold: fine-tune macro-F1 minus frozen-probe macro-F1. Right: folds where fine-tuning wins.
      Only SigLIP backbones lose by fine-tuning. ${srcNote("docs/report/tables/probe_vs_finetune.csv + per-fold records")}</p></div>
    <div class="card"><h3>Training curves (fine-tunes)</h3><div class="ctl"><select id="bcurve"></select></div><div id="bcurvechart"></div>
      <p class="small muted">Validation macro-F1 per epoch on each fold (from <code>history</code> in the run records); ring = best epoch (early-stopping choice).</p></div>
  </div>
  <div class="card"><h3>Per-class F1 (mean over folds)</h3><div id="bheat" class="tablewrap"></div></div>
  <div class="card"><h3>Table</h3><div class="tablewrap scroll"><table class="sortable" id="btab"></table></div></div>`;
  const cols = [["label", "Method"], ["family", "Family"], ["regime", "Regime"], ["schedule", "Schedule"], ["folds", "Folds", 1], ["f1", "Macro-F1", 1], ["f1_sd", "± SD", 1], ["acc", "Accuracy", 1], ["acc_sd", "± SD", 1], ["inscope_acc", "In-scope acc.", 1], ["inscope_f1", "In-scope F1", 1], ["params_M", "Params (M)", 1], ["trainable_M", "Trainable (M)", 1], ["train_min", "Min/fold", 1], ["vram_gb", "VRAM (GB)", 1], ["best_epoch", "Best epoch", 1]];
  function rows() {
    return D.bench.filter((b) => st.fams.includes(b.family) && st.regs.includes(b.regime) &&
      (b.schedule === "-" || (st.sched === "all") || (st.sched === "main" ? b.schedule === "fixed" : b.schedule === "old (bug)")));
  }
  function draw() {
    const rs = rows().filter((b) => b[st.metric] != null).sort((a, b) => b[st.metric] - a[st.metric]);
    const W = 1000, lw = 390, rh = 19, H = rs.length * rh + 40;
    const vals = rs.flatMap((b) => [b[st.metric] - (b[st.metric + "_sd"] || 0), ...(st.metric === "f1" ? b.fold_f1 : st.metric === "acc" ? b.fold_acc : [])]);
    const lo = Math.max(0, Math.floor((Math.min(...vals, 1) - 0.02) * 20) / 20), hi = Math.min(1, Math.ceil((Math.max(...vals, 0) + 0.02) * 20) / 20);
    const x = lin(lo, hi, lw, W - 70);
    let s = svgOpen(W, H, `aria-label="benchmark dot plot"`) + xAxis(x, 0, H - 28, ticks(lo, hi, 8), null, METRICS[st.metric], lw, W - 70);
    rs.forEach((b, i) => {
      const y = i * rh + 10, m = b[st.metric], sd = b[st.metric + "_sd"];
      s += `<text x="${lw - 8}" y="${y + 4}" text-anchor="end" class="lbl">${esc(b.label)}${b.schedule === "old (bug)" ? " [old]" : ""} <tspan class="mut sm">· ${esc(b.regime)}</tspan></text>`;
      if (sd != null) s += `<line x1="${x(m - sd)}" x2="${x(m + sd)}" y1="${y}" y2="${y}" stroke-width="5" stroke-linecap="round" class="${scls(b.group)}" opacity="0.35"/>`;
      const fd = st.metric === "f1" ? b.fold_f1 : st.metric === "acc" ? b.fold_acc : [];
      fd.forEach((v, j) => { s += `<circle cx="${x(v)}" cy="${y + ((j % 2) ? 3 : -3)}" r="2.3" fill="var(--fg2)" opacity="0.65" data-tip="${esc(b.label)} fold ${b.fold_ids[j]}: ${f4(v)}"/>`; });
      s += marker(b.regime, x(m), y, 5.5, gcls(b.group) + " hov", benchTip(b));
      s += `<text x="${x(m + (sd || 0)) + 8}" y="${y + 4}" class="sm">${f3(m)}</text>`;
    });
    $("#bdots", el).innerHTML = rs.length ? s + "</svg>" : "<p class='muted'>No method matches the filters.</p>";
    drawTable(); drawScatter(); drawHeat();
  }
  function drawTable() {
    const rs = rows().slice().sort((a, b) => { const va = a[st.sort], vb = b[st.sort]; if (va == null) return 1; if (vb == null) return -1; return (va < vb ? -1 : va > vb ? 1 : 0) * st.dir; });
    $("#btab", el).innerHTML = `<thead><tr>${cols.map(([k, l, n]) => `<th data-k="${k}" class="${n ? "num" : ""}" tabindex="0" aria-sort="${st.sort === k ? (st.dir > 0 ? "ascending" : "descending") : "none"}">${l}</th>`).join("")}</tr></thead><tbody>` +
      rs.map((b) => `<tr>${cols.map(([k, , n]) => { let v = b[k]; if (k === "family") v = FAM[v].short; if (n && typeof v === "number") v = ["f1", "f1_sd", "acc", "acc_sd", "inscope_acc", "inscope_f1"].includes(k) ? f4(v) : k === "trainable_M" ? (v < 1 ? v.toFixed(4) : v.toFixed(1)) : v; return `<td class="${n ? "num" : ""}">${v == null ? "—" : esc(v)}</td>`; }).join("")}</tr>`).join("") + "</tbody>";
  }
  $("#btab", el).addEventListener("click", (e) => { const th = e.target.closest("th"); if (!th) return; const k = th.dataset.k; st.dir = st.sort === k ? -st.dir : (typeof D.bench[0][k] === "number" ? -1 : 1); st.sort = k; drawTable(); });
  $("#btab", el).addEventListener("keydown", (e) => { if (e.key === "Enter") { const th = e.target.closest("th"); if (th) th.click(); } });
  function drawScatter() {
    const xa = st.xax, rs = rows().filter((b) => b[xa] != null && b[xa] > 0 && b[st.metric] != null);
    const W = 1000, H = 420, l = 70, r = 30, t = 16, bt = 50;
    if (!rs.length) { $("#bscat", el).innerHTML = "<p class='muted'>No method with this cost metric matches the filters.</p>"; return; }
    const xs = rs.map((b) => b[xa]), ys = rs.map((b) => b[st.metric]);
    const xmin = Math.min(...xs) * (st.logx ? 0.6 : 0), xmax = Math.max(...xs) * (st.logx ? 1.6 : 1.08);
    const ylo = Math.max(0, Math.min(...ys) - 0.05), yhi = Math.min(1, Math.max(...ys) + 0.04);
    const x = st.logx ? logS(xmin, xmax, l, W - r) : lin(xmin, xmax, l, W - r), y = lin(ylo, yhi, H - bt, t);
    const lab = { trainable_M: "trainable parameters (millions)", train_min: "fine-tune minutes per fold", params_M: "total parameters (millions)", vram_gb: "peak VRAM (GB)" }[xa];
    let s = svgOpen(W, H, 'aria-label="accuracy vs cost scatter"');
    s += xAxis(x, t, H - bt, st.logx ? logTicks(xmin, xmax) : ticks(xmin, xmax, 8), null, lab, l, W - r) + yAxis(y, l, W - r, ticks(ylo, yhi, 6), null, METRICS[st.metric], t, H - bt);
    if ($("#bpar", el).checked) {
      const srt = rs.slice().sort((a, b) => a[xa] - b[xa] || b[st.metric] - a[st.metric]); let best = -1; const fr = [];
      srt.forEach((b) => { if (b[st.metric] > best) { fr.push(b); best = b[st.metric]; } });
      s += `<path d="${fr.map((b, i) => (i ? `H${x(b[xa])}V${y(b[st.metric])}` : `M${x(b[xa])} ${y(b[st.metric])}`)).join("")}" fill="none" stroke="var(--muted)" stroke-width="1.5" stroke-dasharray="5 3"/>`;
    }
    rs.forEach((b) => {
      const sd = b[st.metric + "_sd"];
      if (sd) s += `<line x1="${x(b[xa])}" x2="${x(b[xa])}" y1="${y(b[st.metric] - sd)}" y2="${y(b[st.metric] + sd)}" class="${scls(b.group)}" opacity="0.4" stroke-width="2"/>`;
      s += marker(b.regime, x(b[xa]), y(b[st.metric]), 6, gcls(b.group) + " hov", benchTip(b));
    });
    const lab2 = rs.slice().sort((a, b) => b[st.metric] - a[st.metric]).filter((b, i, arr) => arr.findIndex((c) => Math.abs(x(c[xa]) - x(b[xa])) < 40 && Math.abs(y(c[st.metric]) - y(b[st.metric])) < 13) === i).slice(0, 14);
    lab2.forEach((b) => { s += `<text x="${x(b[xa]) + 8}" y="${y(b[st.metric]) - 6}" class="sm mut">${esc(b.label.replace(" (partial, last 4 blocks)", " partial").replace(" (partial, 137M trainable)", " partial-old"))}</text>`; });
    $("#bscat", el).innerHTML = s + "</svg>" + legendGroups() + regLegend;
  }
  function drawHeat() {
    const rs = rows().filter((b) => b.per_class_f1).sort((a, b) => b.f1 - a.f1);
    const cw = 34, lw = 290, rh = 15, W = lw + K * cw + 10, H = rs.length * rh + 110;
    let s = svgOpen(W, H, 'aria-label="per-class F1 heatmap"');
    C.forEach((c, j) => { s += `<text transform="translate(${lw + j * cw + cw / 2 + 4},96) rotate(-55)" class="sm">${esc(c)}</text>`; });
    rs.forEach((b, i) => {
      const y = 102 + i * rh;
      s += `<text x="${lw - 6}" y="${y + 11}" text-anchor="end" class="sm">${esc(b.label)}${b.schedule === "old (bug)" ? " [old]" : ""} · ${esc(b.regime)}</text>`;
      b.per_class_f1.forEach((v, j) => { s += `<rect x="${lw + j * cw}" y="${y}" width="${cw - 1}" height="${rh - 1}" fill="hsl(${Math.round(v * 130)} 60% ${62 - v * 22}%)" data-tip="${esc(`${b.label}\n${C[j]}: F1 ${f3(v)}`)}"/>`; });
    });
    $("#bheat", el).innerHTML = s + "</svg><p class='small muted'>Colour: red = F1 0 → green = F1 1. Rows follow the filters above, sorted by macro-F1.</p>";
  }
  // paired view
  (function () {
    const P = D.paired, W = 640, lw = 170, rh = 40, H = P.length * rh + 40, x = lin(-8, 10, lw, W - 70);
    let s = svgOpen(W, H, 'aria-label="probe vs fine-tune"') + xAxis(x, 0, H - 28, ticks(-8, 10, 9), (v) => (v > 0 ? "+" : "") + v, "fine-tune − probe (macro-F1 points)", lw, W - 70);
    s += `<line x1="${x(0)}" x2="${x(0)}" y1="0" y2="${H - 28}" stroke="var(--fg2)"/>`;
    P.forEach((p, i) => {
      const y = i * rh + 18;
      s += `<text x="${lw - 8}" y="${y + 4}" text-anchor="end" class="lbl b">${esc(p.label)}</text><text x="${lw - 8}" y="${y + 17}" text-anchor="end" class="sm mut">${esc(p.type)}</text>`;
      s += `<rect x="${Math.min(x(0), x(p.mean_delta))}" y="${y - 7}" width="${Math.abs(x(p.mean_delta) - x(0))}" height="14" fill="${p.mean_delta > 0 ? "var(--good)" : "var(--bad)"}" opacity="0.25"/>`;
      p.delta.forEach((d, j) => { s += `<circle cx="${x(d)}" cy="${y}" r="4.5" fill="${d > 0 ? "var(--good)" : "var(--bad)"}" data-tip="${esc(`${p.label}, fold ${p.folds[j]}: probe ${f4(p.probe[j])} → fine-tune ${f4(p.ft[j])} (${pts(d, 2)} pts)`)}"/>`; });
      s += `<text x="${W - 64}" y="${y + 4}" class="sm b">${pts(p.mean_delta, 2)} · ${esc(p.wins)}</text>`;
    });
    $("#bpair", el).innerHTML = s + "</svg>";
  })();
  // training curves
  const ftRows = D.bench.filter((b) => b.history);
  $("#bcurve", el).innerHTML = ftRows.map((b, i) => `<option value="${i}">${esc(b.label)} — ${esc(b.schedule)} schedule</option>`).join("");
  function curve(i) { $("#bcurvechart", el).innerHTML = curveSVG(ftRows[i], 560); }
  $("#bcurve", el).addEventListener("change", (e) => curve(+e.target.value)); curve(0);
  onChips(el, "bfam", (v) => { st.fams = v; draw(); }); onChips(el, "breg", (v) => { st.regs = v; draw(); });
  onSeg(el, "bsched", (v) => { st.sched = v; draw(); });
  $("#bmet", el).addEventListener("change", (e) => { st.metric = e.target.value; draw(); });
  $("#bx", el).addEventListener("change", (e) => { st.xax = e.target.value; drawScatter(); });
  $("#blog", el).addEventListener("change", (e) => { st.logx = e.target.checked; drawScatter(); });
  $("#bpar", el).addEventListener("change", drawScatter);
  draw();
  el.__sub = (s, scroll) => { const t = $("#" + s, el); if (t && scroll) setTimeout(() => t.scrollIntoView({ block: "start" }), 30); };
}
function curveSVG(b, W) {
  const H = 260, l = 50, r = 14, t = 12, bt = 40, hs = b.history;
  const ep = Math.max(...hs.flatMap((h) => h.map((e) => e[0])));
  const vs = hs.flatMap((h) => h.map((e) => e[1]));
  const ylo = 0, yhi = Math.min(1, Math.ceil(Math.max(...vs) * 10 + 0.5) / 10);
  const x = lin(0, ep, l, W - r), y = lin(ylo, yhi, H - bt, t);
  let s = svgOpen(W, H, 'aria-label="training curves"') + xAxis(x, t, H - bt, ticks(0, ep, 8), null, "epoch", l, W - r) + yAxis(y, l, W - r, ticks(ylo, yhi, 5), null, "val macro-F1", t, H - bt);
  const cols = ["var(--g-incnn)", "var(--g-vl)", "var(--g-ssl)", "var(--g-invit)", "var(--g-scratch)"];
  hs.forEach((h, k) => {
    s += `<path d="${h.map((e, i) => (i ? "L" : "M") + x(e[0]) + " " + y(e[1])).join("")}" fill="none" stroke="${cols[k % 5]}" stroke-width="1.8"/>`;
    const best = h.reduce((a, e) => (e[1] > a[1] ? e : a));
    s += `<circle cx="${x(best[0])}" cy="${y(best[1])}" r="5" fill="none" stroke="${cols[k % 5]}" stroke-width="2" data-tip="fold ${b.fold_ids[k]}: best epoch ${best[0]}, val macro-F1 ${f4(best[1])}"/>`;
  });
  s += `<text x="${W - r}" y="${t + 10}" text-anchor="end" class="sm mut">${b.fold_ids.map((f, k) => `fold ${f}`).join(" · ")}</text>`;
  return s + "</svg>" + `<div class="legend">${b.fold_ids.map((f, k) => `<span><span class="dot" style="background:${cols[k % 5]}"></span>fold ${f}</span>`).join("")}</div>`;
}

// ==================================================================================================== 5. ABLATIONS
function renderAblations(el) {
  const S = D.schedule, T = D.tta, CL = D.clean, t = D.final.test, w = D.final.web, F = D.fit;
  const part = D.bench.filter((b) => b.name.startsWith("siglip_so400m_384_partial"));
  const probe012 = D.kn.partial_ft.probe_folds012_f1;
  const card = (q, verdict, vcls, body, src) => `<div class="card"><h3>${q} <span class="badge ${vcls}">${verdict}</span></h3>${body}${srcNote(src)}</div>`;
  // schedule chart
  const sch = (function () {
    const W = 600, lw = 130, rh = 24, H = S.rows.length * rh + 40, x = lin(-4, 4, lw, W - 105);
    let s = svgOpen(W, H, 'aria-label="schedule ablation"') + xAxis(x, 0, H - 28, ticks(-4, 4, 8), (v) => (v > 0 ? "+" : "") + v, "fixed − old schedule (macro-F1 points), folds 0–2", lw, W - 105);
    s += `<rect x="${x(-S.noise2se)}" y="0" width="${x(S.noise2se) - x(-S.noise2se)}" height="${H - 28}" fill="var(--muted)" opacity="0.15" data-tip="±${S.noise2se} pt noise floor (2 SE of a 3-fold mean, from the SD of all ${S.n} per-fold differences)"/>`;
    s += `<line x1="${x(0)}" x2="${x(0)}" y1="0" y2="${H - 28}" stroke="var(--fg2)"/>`;
    S.rows.forEach((r, i) => {
      const y = i * rh + 14;
      s += `<text x="${lw - 8}" y="${y + 4}" text-anchor="end" class="lbl">${esc(r.label)}</text>`;
      s += `<line x1="${x(0)}" x2="${x(r.mean)}" y1="${y}" y2="${y}" stroke="${r.mean > 0 ? "var(--good)" : "var(--bad)"}" stroke-width="6" opacity="0.35"/>`;
      r.delta.forEach((d, j) => { s += `<circle cx="${x(d)}" cy="${y}" r="4" fill="${d > 0 ? "var(--good)" : "var(--bad)"}" data-tip="${esc(`${r.label} fold ${j}: ${pts(d, 2)} pts`)}"/>`; });
      s += `<text x="${W - 88}" y="${y + 4}" class="sm">${pts(r.mean, 2)} · ${r.improved}/3</text>`;
    });
    return s + "</svg>";
  })();
  const fusionGrid = D.fusion_grid.filter((r) => r.T === 50);
  el.innerHTML = `<h2>Ablations — each a question with an honest answer</h2>
  <p class="lead">Several ablations are <b>null results</b>: the effect is smaller than the noise floor of a ~2.5k-image dataset. They are
  reported as such; the badge says what the evidence supports.</p>
  <div class="grid g2">
  ${card("Should the pretrained backbone be frozen or fine-tuned?", "depends on backbone", "mixed",
    `<p>Paired on identical folds (fine-tune − probe): ${D.paired.map((p) => `${esc(p.label)} <b>${pts(p.mean_delta, 2)}</b> (${p.wins})`).join("; ")}.</p>
     <p><b>Interpretation.</b> Fine-tuning helps when the frozen feature is mediocre (ImageNet-1k ResNet, ConvNeXt, DINOv2, CLIP-B) and
     hurts or ties when it is already excellent (SigLIP). The finding "frozen beats fine-tuned" is therefore <b>SigLIP-specific</b>.
     Fine-tunes early-stop on the scored fold, which favours them — so the SigLIP result is conservative. One hand-set recipe per family; a
     per-family search could narrow the gaps. <a href="#bench" data-go="bench">See the paired chart →</a></p>`, "docs/report/tables/probe_vs_finetune.csv")}
  ${card("Can a partial fine-tune of the best backbone beat its probe?", "no", "null",
    `<table><tr><th>Variant (folds 0–2)</th><th class="num">Trainable</th><th class="num">Min/fold</th><th class="num">Macro-F1</th></tr>
     <tr><td>Frozen probe</td><td class="num">${(20 * 1153 / 1000).toFixed(1)}k</td><td class="num">—</td><td class="num">${f4(probe012)}</td></tr>
     ${part.map((b) => `<tr><td>${esc(b.label)}${b.schedule === "old (bug)" ? " — old schedule" : ""}</td><td class="num">${b.trainable_M.toFixed(1)}M</td><td class="num">${b.train_min}</td><td class="num">${f4(b.f1)} ± ${f4(b.f1_sd)}</td></tr>`).join("")}</table>
     <p><b>Interpretation.</b> Neither variant beats the probe (differences within fold noise), at 17–23 min per fold. The final model keeps the backbone frozen.</p>`, "results/benchmark.csv")}
  ${card("Did the learning-rate schedule fix matter?", "null (confounded)", "null",
    `${sch}<p>${S.n_improved}/${S.n} fold scores improved; mean ${pts(S.mean, 2)} pts; SD of per-fold differences ${S.sd} pts; exact sign test p = ${S.p}.</p>
     <p><b>Interpretation.</b> No consistent effect. Only the BatchNorm CNNs (ResNet-50, EfficientNetV2-S, MobileNetV3-L) move up consistently —
     consistent with the <i>BatchNorm-freezing</i> change that shipped in the same commit (<code>d6d4e09</code>), not necessarily the warm-up. The two are confounded.</p>`, "results/schedule_ablation_folds012.csv, results/phase2/raw_oldschedule/runs_{A,B}.jsonl, results/phase2/runs_fixed.jsonl")}
  ${card("Does horizontal-flip test-time augmentation help the probe?", "mixed / below 1 pt", "mixed",
    `<table><tr><th>Probe</th><th class="num">no TTA</th><th class="num">flip TTA</th><th>Per-fold Δ (pts)</th></tr>${T.map((r) => `<tr><td>${esc(r.label)}</td><td class="num">${f4(r.base)}</td><td class="num">${f4(r.tta)}</td><td>${r.delta.map((d) => `<span class="${d > 0 ? "good" : "bad"}">${pts(d, 2)}</span>`).join(" ")}</td></tr>`).join("")}</table>
     <p><b>Interpretation.</b> Opposite signs on two near-identical backbones and under one point either way; TTA doubles inference cost, so it was not adopted.</p>`, "results/phase1/raw/runs.jsonl, results/phase1b/runs.jsonl")}
  ${card("Does cleaning the training data help?", "no — dropping OOD rows hurts", "hurts",
    `${barsH(CL.map((r) => ({ label: r.variant, v: r.in_acc, text: `in-scope ${f4(r.in_acc)} · all ${f4(r.all_acc)}`, cls: r.variant === "keep_all" ? "g-Vision-language" : "g-Classical", tip: `${r.variant}: in-scope acc ${r.in_acc}, all-row acc ${r.all_acc}, all-row F1 ${r.all_f1}` })), { w: 600, lw: 120, rw: 190, min: 0.84, max: 0.87, nt: 3 })}
     <p>SigLIP So400m probe, rows removed from <i>training</i> only; always scored on all out-of-fold rows. Dropping the 69 label issues: ${pts(100 * (CL[1].in_acc - CL[0].in_acc), 2)} pts in-scope (noise).
     Dropping OOD-flagged rows: ${pts(100 * (CL[2].in_acc - CL[0].in_acc), 2)} pts in-scope and ${pts(100 * (CL[2].all_acc - CL[0].all_acc), 1)} pts on all rows.</p>
     <p><b>Interpretation.</b> Many flagged rows still carry class signal (or match flagged evaluation images). Training data were left unchanged.</p>`, "results/phase1/cleaning_ablation_siglip_so400m.csv")}
  ${card("Does fusing the backbone's zero-shot head help?", "null in-distribution · helps out-of-source", "mixed",
    `<table><tr><th>Evaluation</th><th class="num">probe only</th><th class="num">fused</th><th class="num">Δ (pts)</th></tr>
     <tr><td>Dev OOF (n = ${F.n_dev})</td><td class="num">${f4(F.oof_probe_only_accuracy)}</td><td class="num">${f4(F.oof_accuracy)}</td><td class="num">${pts(100 * (F.oof_accuracy - F.oof_probe_only_accuracy), 2)}</td></tr>
     <tr><td>Sealed test (n = ${t.n})</td><td class="num">${f4(t.probe_only_accuracy)}</td><td class="num">${f4(t.accuracy)}</td><td class="num">${pts(100 * (t.accuracy - t.probe_only_accuracy), 2)}</td></tr>
     <tr><td>Web holdout (n = ${w.n})</td><td class="num">${f4(w.probe_only_accuracy)}</td><td class="num">${f4(w.accuracy)}</td><td class="num"><b>${pts(100 * (w.accuracy - w.probe_only_accuracy), 2)}</b></td></tr></table>
     <p>Fusion = log p_probe + ${F.w_zs} · log softmax(${F.t_zs} · cos), weight and temperature fixed a priori. An earlier Phase-1 grid (zero-shot scores from open_clip's slightly different preprocessing) had suggested a larger gain:
     ${fusionGrid.map((r) => `w = ${r.w}: ${f4(r.acc)}`).join(", ")} (<code>results/phase1/fusion_ablation.csv</code>) — it did not survive the single deployable pipeline.</p>
     <p><b>Interpretation.</b> In-distribution the effect is within ±${D.kn.test_accuracy_se_pts} pt SE (a tie). On out-of-source photos the text head adds robustness, consistent with reports that keeping a
     vision-language model's zero-shot classifier preserves robustness (WiSE-FT). Per-image probe-only predictions were not saved, so no paired McNemar test exists.</p>`, "exports/final/fit_report.json, results/final/final_test_v2.json, results/final/final_test_web_holdout_v1.json")}
  ${card("How much did the legacy split's leakage inflate scores?", "large effect", "hurts",
    `<p>${D.leakage.rows.map((r) => `${esc(r.model)}: ${f4(r.grouped_cv_acc)} → ${f4(r.legacy_style_acc)} (<b>+${r.inflation_pts.toFixed(1)}</b> pts)`).join("<br>")}</p>
     <p><b>Interpretation.</b> The legacy numbers (0.71–0.76) are not comparable with v2. Weaker models gain most, so leakage also compresses family gaps. Accuracy on the
     non-duplicated legacy rows is still above grouped-CV accuracy (e.g. ${f4(D.leakage.rows.find((r) => r.model === "siglip_so400m_384").acc_on_clean)} vs ${f4(D.leakage.rows.find((r) => r.model === "siglip_so400m_384").grouped_cv_acc)} for SigLIP) — likely near-duplicates the count misses; not decomposed further.
     <a href="#protocol" data-go="protocol">Protocol →</a></p>`, "results/leakage_simulation.csv")}
  ${card("Does more training compute buy accuracy?", "not across families", "null",
    `<p>Among fine-tunes, accuracy follows the <b>pretraining source</b>: MobileNetV3-L trains in ${D.bench.find((b) => b.name === "mobilenetv3_large" && b.schedule === "fixed").train_min} min/fold and ResNet-50 in ${D.bench.find((b) => b.name === "resnet50" && b.schedule === "fixed").train_min},
     while the So400m partial fine-tune needs ${D.bench.find((b) => b.name === "siglip_so400m_384_partial_v2").train_min} min/fold and still ties its probe. On the trainable-parameter axis the Pareto front is almost entirely linear probes.</p>
     <p><b>Interpretation.</b> The relevant deployment cost is the frozen backbone's inference, not training. <a href="#bench/scatter" data-go="bench/scatter">Scatter →</a></p>`, "results/benchmark.csv")}
  </div>`;
}
// ==================================================================================================== diagram helpers
function dia(W, H, aria) { const id = uid("ah"); return { id, s: `<svg viewBox="0 0 ${W} ${H}" width="${W}" class="dia" role="img" aria-label="${esc(aria)}"><defs><marker id="${id}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" class="ah"/></marker></defs>` }; }
function bx(x, y, w, h, label, cls, sub, tipv) {
  const lines = String(label).split("\n");
  let s = `<rect class="box ${cls || ""}" x="${x}" y="${y}" width="${w}" height="${h}" rx="7" ${tipv ? `data-tip="${esc(tipv)}"` : ""}/>`;
  const n = lines.length + (sub ? 1 : 0), y0 = y + h / 2 - (n - 1) * 7.5 + 4;
  lines.forEach((l, i) => { s += `<text x="${x + w / 2}" y="${y0 + i * 15}" text-anchor="middle" class="t">${esc(l)}</text>`; });
  if (sub) s += `<text x="${x + w / 2}" y="${y0 + lines.length * 15}" text-anchor="middle" class="sm mut">${esc(sub)}</text>`;
  return s;
}
function ar(id, x1, y1, x2, y2, cls) { return `<line class="ln ${cls || ""}" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" marker-end="url(#${id})"/>`; }
function pathA(id, d, cls) { return `<path class="ln ${cls || ""}" d="${d}" marker-end="url(#${id})"/>`; }
function tx(x, y, s, cls, anchor) { return `<text x="${x}" y="${y}" class="${cls || ""}" text-anchor="${anchor || "start"}">${esc(s)}</text>`; }

function finalDiagram() {
  const P = D.pre, F = D.fit, d = dia(1000, 300, "final model data flow");
  let s = d.s;
  s += bx(6, 110, 96, 70, "image", "img", "any size, RGB");
  s += ar(d.id, 102, 145, 124, 145);
  s += bx(124, 105, 168, 80, "preprocess", "m", `short ${P.resize_short_side} → crop ${P.input_size}`);
  s += tx(208, 200, `mean = std = ${P.mean[0]}, bicubic`, "sm mut", "middle");
  s += ar(d.id, 292, 145, 316, 145);
  s += bx(316, 80, 172, 130, "SigLIP So400m/14\nimage tower", "a", "27 blocks · frozen", "patch 14 → 27×27 = 729 tokens × 1152; MAP attention pooling");
  s += tx(402, 228, "3×384×384 → 729×1152 → 1152", "sm mut", "middle");
  s += ar(d.id, 488, 145, 528, 145);
  s += tx(510, 136, "f", "sm b", "middle");
  s += pathA(d.id, "M532 145 V70 H566");
  s += pathA(d.id, "M532 145 V220 H566");
  s += bx(566, 35, 184, 70, "probe head", "g", `standardise → LR (C = ${F.final_C})`, "logistic regression refit on all dev rows; log-softmax");
  s += bx(566, 185, 184, 70, "zero-shot head", "p", `log softmax(${F.t_zs}·cos(f, text))`, "cosine to template-averaged class text embeddings (1152 × 20)");
  s += pathA(d.id, "M750 70 H780 V128");
  s += pathA(d.id, "M750 220 H780 V162");
  s += tx(786, 200, `× ${F.w_zs}`, "sm b", "start");
  s += `<circle cx="780" cy="145" r="17" class="box o" style="fill:var(--card);stroke:var(--warn)"/>` + tx(780, 150, "+", "t", "middle");
  s += ar(d.id, 797, 145, 818, 145);
  s += bx(818, 110, 76, 70, "÷ T", "o", `T = ${(+F.temperature).toFixed(4)}`, "temperature scaling fit on dev out-of-fold scores");
  s += ar(d.id, 894, 145, 910, 145);
  s += bx(910, 95, 86, 100, "softmax\n→ max p", "m", `≥ ${F.abstain_threshold} ?`, "answer if max probability ≥ threshold, else UNSURE");
  s += tx(953, 215, "class / UNSURE", "sm b", "middle");
  s += tx(500, 290, "One ONNX graph (image tower + both heads + temperature); outputs logits and probs. labels.json + preprocess.json alongside.", "sm mut", "middle");
  return s + "</svg>";
}

// ==================================================================================================== 6. FINAL MODEL
function predsOf(ds) {
  if (ds.__p) return ds.__p;
  const pred = [], conf = [];
  ds.probs.forEach((p) => { let m = 0; for (let j = 1; j < p.length; j++) if (p[j] > p[m]) m = j; pred.push(m); conf.push(p[m]); });
  return (ds.__p = { pred, conf });
}
function selective(ds, thr) {
  const { pred, conf } = predsOf(ds); let a = 0, c = 0;
  for (let i = 0; i < ds.n; i++) if (conf[i] >= thr) { a++; if (pred[i] === ds.y[i]) c++; }
  return { cov: a / ds.n, acc: a ? c / a : null, answered: a, correct: c };
}
function eceOf(ds, bins) {
  const { pred, conf } = predsOf(ds), out = []; let e = 0;
  for (let b = 0; b < bins; b++) {
    const lo = b / bins, hi = (b + 1) / bins; let n = 0, cs = 0, ac = 0;
    for (let i = 0; i < ds.n; i++) if (conf[i] > lo && conf[i] <= hi) { n++; cs += conf[i]; if (pred[i] === ds.y[i]) ac++; }
    out.push({ lo, hi, n, conf: n ? cs / n : null, acc: n ? ac / n : null });
    if (n) e += (n / ds.n) * Math.abs(ac / n - cs / n);
  }
  return { bins: out, ece: e };
}
function dsThumb(ds, i) { return ds === D.final.web ? "w" + ds.ids[i] : ds.ids[i]; }
function dsCredit(ds, i) { const c = ds.credit && ds.credit[i]; return c ? `\n© ${c[0]} · ${c[1]}\n${c[2]}` : ""; }
function renderFinal(el) {
  const st = { ds: "test", norm: false, abst: false, thr: D.fit.abstain_threshold, bins: 15, cell: null, row: null };
  const DS = () => D.final[st.ds];
  el.innerHTML = `<h2>Final model: frozen SigLIP So400m + fused heads + calibrated abstention</h2>
  <div class="card"><h3>Architecture and data flow</h3>${finalDiagram()}<p class="diacap">Hover the boxes for details. Shapes from timm 1.0.30
  (<code>vit_so400m_patch14_siglip_384</code>) and <code>exports/final/preprocess.json</code>; code: <code>src/compclass/final.py</code>.
  The <a href="#families/siglip" data-go="families/siglip">SigLIP family page</a> explains the backbone.</p></div>
  <div class="grid g12">
    <div class="card"><h3>Model card</h3><table>${D.model_card.map((r) => `<tr><th style="position:static">${esc(r.Field)}</th><td>${esc(r.Value)}</td></tr>`).join("")}</table>${srcNote("docs/report/tables/model_card.csv")}</div>
    <div class="card"><h3>Abstention: how many images to answer?</h3>
      <div class="ctl"><label for="thr">confidence threshold</label><input type="range" id="thr" min="0" max="0.99" step="0.01" value="${st.thr}"><b id="thrv"></b>
      <button class="btn" type="button" id="thr05">chosen: ${D.fit.abstain_threshold}</button></div>
      <div id="abstats"></div><div id="abcurve"></div>
      <p class="small muted">Recomputed live from the embedded per-image probabilities (<code>results/final/*_preds.npz</code>, rounded to 5 decimals).
      Dev OOF points are the grid in <code>exports/final/fit_report.json</code>; the threshold ${D.fit.abstain_threshold} was chosen there (smallest grid value with OOF accepted accuracy ≥ 0.90) before the test was read.</p></div>
  </div>
  <div class="card"><h3>Confusion matrix</h3>
    <div class="ctl">${seg("cmds", [["test", `Sealed test (n = ${D.final.test.n})`], ["web", `Web holdout (n = ${D.final.web.n})`]], "test")}
      ${seg("cmnorm", [["0", "counts"], ["1", "row-normalised"]], "0")}
      <label><input type="checkbox" id="cmab"> only answered images (threshold above)</label></div>
    <div class="grid g21"><div id="cm" class="tablewrap"></div><div id="cmside"></div></div>
    <p class="small muted">Rows = true class, columns = prediction. Hover a cell; click a cell to see its images; click a row label for that class's precision/recall.</p></div>
  <div class="grid g2">
    <div class="card"><h3>Calibration (reliability diagram)</h3><div class="ctl"><label for="bins">bins</label><input type="range" id="bins" min="5" max="25" step="1" value="15"><b id="binsv"></b></div>
      <div id="rel"></div></div>
    <div class="card"><h3>Mistakes worth looking at</h3><div class="ctl">${seg("mist", [["conf", "most confident mistakes"], ["unsure", "least confident (UNSURE)"]], "conf")}</div><div id="mistg" class="gallery"></div></div>
  </div>
  <div class="card"><h3>Deployment</h3><div class="grid g3">
    <div><div class="stat"><b>${(D.export.onnx_mb / 1000).toFixed(2)} GB</b></div><div class="small">ONNX graph (external-data format)</div></div>
    <div><div class="stat"><b>${D.export.ort_cpu_ms_per_image} ms</b></div><div class="small">per image, onnxruntime CPU; max |Δp| vs PyTorch ${D.export.ort_max_abs_prob_diff.toExponential(1)}, argmax agreement ${pct(D.export.ort_argmax_agree, 0)}</div></div>
    <div><div class="stat"><b>${D.demo_parity.n_images}/${D.demo_parity.n_images}</b></div><div class="small">demo parity images (preprocessing + ONNX vs PyTorch), argmax agreement ${pct(D.demo_parity.argmax_agreement, 0)}</div></div></div>
    <div class="callout warn small">OpenCV's DNN importer crashes on this graph (<code>${esc(D.export.cv2_error)}</code> — segmentation fault), so the demo uses OpenCV only for capture/drawing and onnxruntime for inference. Real-time CPU use would need a smaller distilled model.</div>
    ${srcNote("exports/final/export_report.json, results/demo/demo_parity.json")}</div>`;
  // abstention
  const oof = D.fit.abstain_table;
  function abst() {
    $("#thrv", el).textContent = st.thr.toFixed(2);
    const a = selective(D.final.test, st.thr), b = selective(D.final.web, st.thr);
    const row = (nm, ds, r) => `<tr><td>${nm}</td><td class="num">${r.answered}/${ds.n}</td><td class="num"><b>${pct(r.cov)}</b></td><td class="num"><b>${r.acc == null ? "—" : pct(r.acc)}</b></td><td class="num">${ds.n - r.answered}</td><td class="num">${r.answered - r.correct}</td></tr>`;
    $("#abstats", el).innerHTML = `<table><tr><th>Set</th><th class="num">answered</th><th class="num">coverage</th><th class="num">accepted acc.</th><th class="num">UNSURE</th><th class="num">errors left</th></tr>${row("Sealed test", D.final.test, a)}${row("Web holdout", D.final.web, b)}</table>`;
    const W = 560, H = 270, l = 52, r = 14, t = 12, bt = 40, x = lin(0.4, 1, l, W - r), y = lin(0.8, 1, H - bt, t);
    let s = svgOpen(W, H, 'aria-label="coverage vs accepted accuracy"') + xAxis(x, t, H - bt, ticks(0.4, 1, 6), (v) => Math.round(v * 100) + "%", "coverage (fraction answered)", l, W - r) + yAxis(y, l, W - r, ticks(0.8, 1, 4), (v) => Math.round(v * 100) + "%", "accuracy on answered", t, H - bt);
    const grid = []; for (let v = 0; v <= 0.99 + 1e-9; v += 0.01) grid.push(+v.toFixed(2));
    [["test", "var(--accent)"], ["web", "var(--g-invit)"]].forEach(([k, col]) => {
      const pts_ = grid.map((g) => selective(D.final[k], g)).filter((p) => p.acc != null && p.cov >= 0.4);
      s += `<path d="${pts_.map((p, i) => (i ? "L" : "M") + x(p.cov) + " " + y(Math.max(0.8, p.acc))).join("")}" fill="none" stroke="${col}" stroke-width="2"/>`;
      const c = selective(D.final[k], st.thr);
      if (c.acc != null && c.cov >= 0.4) s += `<circle cx="${x(c.cov)}" cy="${y(Math.max(0.8, c.acc))}" r="6" fill="${col}" data-tip="${k === "test" ? "sealed test" : "web holdout"} at ${st.thr.toFixed(2)}: coverage ${pct(c.cov)}, accepted ${pct(c.acc)}"/>`;
    });
    oof.filter((o) => o.accepted_acc != null && o.coverage >= 0.4).forEach((o) => { s += `<rect x="${x(o.coverage) - 3}" y="${y(Math.max(0.8, o.accepted_acc)) - 3}" width="6" height="6" fill="var(--g-ssl)" data-tip="dev OOF, threshold ${o.threshold}: coverage ${pct(o.coverage)}, accepted ${pct(o.accepted_acc)}"/>`; });
    $("#abcurve", el).innerHTML = s + `</svg><div class="legend"><span><span class="dot" style="background:var(--accent)"></span>sealed test</span><span><span class="dot" style="background:var(--g-invit)"></span>web holdout</span><span><span class="dot" style="background:var(--g-ssl);border-radius:0"></span>dev OOF grid</span></div>`;
    if (st.abst) cm();
  }
  $("#thr", el).addEventListener("input", (e) => { st.thr = +e.target.value; abst(); });
  $("#thr05", el).addEventListener("click", () => { st.thr = D.fit.abstain_threshold; $("#thr", el).value = st.thr; abst(); });
  // confusion
  function cm() {
    const ds = DS(), { pred, conf } = predsOf(ds), M = C.map(() => C.map(() => []));
    let unsure = 0;
    for (let i = 0; i < ds.n; i++) { if (st.abst && conf[i] < st.thr) { unsure++; continue; } M[ds.y[i]][pred[i]].push(i); }
    const cw = 25, lw = 150, top = 120, W = lw + K * cw + 60, H = top + K * cw + 10;
    let s = svgOpen(W, H, 'aria-label="confusion matrix"');
    C.forEach((c, j) => { s += `<text transform="translate(${lw + j * cw + cw / 2 + 4},${top - 6}) rotate(-55)" class="sm">${esc(c)}</text>`; });
    C.forEach((c, i) => {
      const tot = M[i].reduce((a, r) => a + r.length, 0);
      s += `<text x="${lw - 6}" y="${top + i * cw + cw / 2 + 4}" text-anchor="end" class="lbl clk ${st.row === i ? "b" : ""}" data-row="${i}" tabindex="0">${esc(c)}</text>`;
      C.forEach((c2, j) => {
        const n = M[i][j].length, v = st.norm ? (tot ? n / tot : 0) : n, vmax = st.norm ? 1 : 12, a = Math.min(1, v / vmax);
        const fill = i === j ? `color-mix(in srgb, var(--good) ${Math.round(a * 85)}%, var(--cell0))` : `color-mix(in srgb, var(--bad) ${Math.round(Math.min(1, a * 3) * 80)}%, var(--cell0))`;
        const sel = st.cell && st.cell[0] === i && st.cell[1] === j;
        s += `<rect x="${lw + j * cw}" y="${top + i * cw}" width="${cw - 1}" height="${cw - 1}" style="fill:${fill}" ${sel ? 'stroke="var(--accent)" stroke-width="2.5"' : ""} class="clk" data-cell="${i},${j}" data-tip="${esc(`true ${c} → predicted ${c2}: ${n} image${n === 1 ? "" : "s"}${tot ? ` (${pct(n / tot, 0)} of the row)` : ""}`)}"/>`;
        if (n) s += `<text x="${lw + j * cw + cw / 2}" y="${top + i * cw + cw / 2 + 4}" text-anchor="middle" class="sm" pointer-events="none">${st.norm ? Math.round(100 * v) : n}</text>`;
      });
      s += `<text x="${lw + K * cw + 6}" y="${top + i * cw + cw / 2 + 4}" class="sm mut">${tot}</text>`;
    });
    $("#cm", el).innerHTML = s + "</svg>" + (st.abst ? `<p class="small">${unsure} images abstained (UNSURE) at threshold ${st.thr.toFixed(2)} and are not in the matrix.</p>` : "");
    side(M);
  }
  function side(M) {
    const ds = DS(), box = $("#cmside", el);
    if (st.cell) {
      const [i, j] = st.cell, ids = M[i][j], { conf } = predsOf(ds);
      box.innerHTML = `<h4>true <b>${esc(C[i])}</b> → predicted <b>${esc(C[j])}</b>: ${ids.length}</h4><div class="gallery">${ids.slice(0, 40).map((k) => thumb(dsThumb(ds, k), `p = ${conf[k].toFixed(2)}${ds.ood[k] ? " · OOD-flag" : ""}`, `${ds.ids[k]}: true ${C[ds.y[k]]}, predicted ${C[predsOf(ds).pred[k]]} (p=${conf[k].toFixed(3)})` + dsCredit(ds, k), i === j ? "good" : "bad")).join("")}</div>${ids.length > 40 ? `<p class="small muted">first 40 of ${ids.length}</p>` : ""}`;
      return;
    }
    const r = st.row != null ? st.row : null;
    if (r == null) { box.innerHTML = `<p class="muted">Click a cell or a row label.</p><p class="small">Weakest class on the sealed test: <b>resistor</b> (recall ${f3(D.final.test.per_class.resistor.recall)}, precision ${f3(D.final.test.per_class.resistor.precision)}). On the web holdout: ${Object.entries(D.final.web.per_class).sort((a, b) => a[1].recall - b[1].recall).slice(0, 3).map(([c, v]) => `${esc(c)} (${f3(v.recall)})`).join(", ")}.</p>`; return; }
    const pc = ds.per_class[C[r]], tot = M[r].reduce((a, x) => a + x.length, 0), tp = M[r][r].length, ci = wilson(tp, tot);
    const outs = M[r].map((x, j) => [j, x.length]).filter(([j, n]) => j !== r && n).sort((a, b) => b[1] - a[1]);
    box.innerHTML = `<h4>${esc(C[r])}</h4><table><tr><td>support</td><td class="num">${pc.support}</td></tr><tr><td>precision</td><td class="num">${f3(pc.precision)}</td></tr><tr><td>recall</td><td class="num">${f3(pc.recall)}</td></tr><tr><td>F1</td><td class="num">${f3(pc.f1)}</td></tr>${st.abst ? "" : `<tr><td>recall 95 % Wilson CI</td><td class="num">${f2(ci[0])}–${f2(ci[1])}</td></tr>`}</table>
      <p class="small">${outs.length ? "Misrouted to: " + outs.map(([j, n]) => `${esc(C[j])} ${n}`).join(", ") : "No errors in this row."}</p><p class="tiny muted">Precision/recall/F1 from <code>${esc(ds.source)}</code>${st.abst ? " (all images, without abstention)" : ""}.</p>`;
  }
  $("#cm", el).addEventListener("click", (e) => {
    const c = e.target.closest("[data-cell]"), r = e.target.closest("[data-row]");
    if (c) { const [i, j] = c.dataset.cell.split(",").map(Number); st.cell = st.cell && st.cell[0] === i && st.cell[1] === j ? null : [i, j]; st.row = null; cm(); }
    else if (r) { st.row = +r.dataset.row; st.cell = null; cm(); }
  });
  $("#cm", el).addEventListener("keydown", (e) => { if (e.key === "Enter") { const r = e.target.closest("[data-row]"); if (r) { st.row = +r.dataset.row; st.cell = null; cm(); } } });
  onSeg(el, "cmds", (v) => { st.ds = v; st.cell = null; cm(); rel(); mist(); });
  onSeg(el, "cmnorm", (v) => { st.norm = v === "1"; cm(); });
  $("#cmab", el).addEventListener("change", (e) => { st.abst = e.target.checked; cm(); });
  // reliability
  function rel() {
    $("#binsv", el).textContent = st.bins;
    const ds = DS(), R = eceOf(ds, st.bins), W = 520, H = 300, l = 48, r = 14, t = 12, bt = 42, x = lin(0, 1, l, W - r), y = lin(0, 1, H - bt, t);
    let s = svgOpen(W, H, 'aria-label="reliability diagram"') + xAxis(x, t, H - bt, ticks(0, 1, 5), null, "confidence (max probability)", l, W - r) + yAxis(y, l, W - r, ticks(0, 1, 5), null, "accuracy", t, H - bt);
    s += `<line x1="${x(0)}" y1="${y(0)}" x2="${x(1)}" y2="${y(1)}" stroke="var(--muted)" stroke-dasharray="4 3"/>`;
    R.bins.forEach((b) => {
      if (!b.n) return;
      s += `<rect x="${x(b.lo) + 1}" y="${y(b.acc)}" width="${x(b.hi) - x(b.lo) - 2}" height="${y(0) - y(b.acc)}" fill="var(--accent)" opacity="0.55" data-tip="${esc(`confidence (${b.lo.toFixed(2)}, ${b.hi.toFixed(2)}]: ${b.n} images, accuracy ${f3(b.acc)}, mean confidence ${f3(b.conf)}`)}"/>`;
      s += `<circle cx="${x(b.conf)}" cy="${y(b.conf)}" r="3.5" fill="var(--fg)"/>`;
      s += `<text x="${(x(b.lo) + x(b.hi)) / 2}" y="${y(b.acc) - 4}" text-anchor="middle" class="sm mut">${b.n}</text>`;
    });
    $("#rel", el).innerHTML = s + `</svg><p class="small">ECE (${st.bins} equal-width bins) = <b>${R.ece.toFixed(4)}</b>${st.bins === 15 ? ` — file value ${f4(ds.ece)}` : ""}. Bar = accuracy, dot = mean confidence, number = images in bin; bins with few images are noisy.
      Temperature T = ${(+D.fit.temperature).toFixed(4)} was fit on dev OOF scores (OOF ECE ${f4(D.fit.ece_before)} → ${f4(D.fit.ece_after)}).</p>`;
  }
  $("#bins", el).addEventListener("input", (e) => { st.bins = +e.target.value; rel(); });
  let mmode = "conf";
  function mist() {
    const ds = DS(), { pred, conf } = predsOf(ds);
    let idx = [...Array(ds.n).keys()];
    if (mmode === "conf") idx = idx.filter((i) => pred[i] !== ds.y[i]).sort((a, b) => conf[b] - conf[a]).slice(0, 18);
    else idx = idx.sort((a, b) => conf[a] - conf[b]).slice(0, 18);
    $("#mistg", el).innerHTML = idx.map((i) => thumb(dsThumb(ds, i), `true <b>${esc(C[ds.y[i]])}</b><br>→ ${esc(C[pred[i]])} (${conf[i].toFixed(2)})${ds.ood[i] ? "<br><span class='warn'>OOD-flagged</span>" : ""}`, `${ds.ids[i]}` + dsCredit(ds, i), pred[i] === ds.y[i] ? (conf[i] < st.thr ? "uns" : "good") : "bad")).join("");
  }
  onSeg(el, "mist", (v) => { mmode = v; mist(); });
  abst(); cm(); rel(); mist();
}

// ==================================================================================================== 7. GENERALISATION
function renderGeneral(el) {
  const t = D.final.test, w = D.final.web, k = D.kaggle, rp = k.replay;
  const rowsT = [["Accuracy (95 % Wilson CI)", `${f3(t.accuracy)} (${f3(t.ci[0])}–${f3(t.ci[1])})`, `${f3(w.accuracy)} (${f3(w.ci[0])}–${f3(w.ci[1])})`],
    ["Macro-F1", f3(t.macro_f1), f3(w.macro_f1)], ["Probe-only accuracy (no zero-shot head)", f3(t.probe_only_accuracy), f3(w.probe_only_accuracy)],
    [`Coverage at threshold ${D.fit.abstain_threshold}`, f3(t.coverage), f3(w.coverage)], ["Accuracy on answered images", f3(t.accepted_accuracy), f3(w.accepted_accuracy)],
    ["ECE (15 bins)", f3(t.ece), f3(w.ece)], ["In-scope accuracy", `${f3(t.inscope_accuracy)} (n = ${t.inscope_n})`, "— (no OOD flags)"]];
  el.innerHTML = `<h2>Generalisation: two sealed sets and the original Kaggle files</h2>
  <div class="grid g2">
    <div class="card"><h3>Final model on the two sealed sets</h3><table><tr><th>Metric</th><th class="num">Kaggle sealed test (n = ${t.n})</th><th class="num">Web holdout (n = ${w.n})</th></tr>${rowsT.map((r) => `<tr><td>${r[0]}</td><td class="num">${r[1]}</td><td class="num">${r[2]}</td></tr>`).join("")}</table>
      <ul class="small"><li><b>Out-of-source accuracy is close to in-distribution accuracy</b>; the intervals overlap heavily.</li>
      <li><b>The zero-shot head pays off under shift</b>: ${pts(100 * (w.accuracy - w.probe_only_accuracy))} pts over the probe on web photos, ${pts(100 * (t.accuracy - t.probe_only_accuracy))} in-distribution (a tie).</li>
      <li>Web holdout: ${esc(D.web_meta.source)}, fingerprint <code>${esc(w.fingerprint)}</code>, CLIP leakage gate against every v2 image, evaluated once.</li></ul>
      <div class="callout warn small">Caveat: the holdout was curated by one automated reviewer who kept only clear, single-component photos — easier than an unbiased web sample. ${(() => { const v = Object.values(D.web_meta.counts), m = Math.min(...v); return `${v.filter((x) => x === m).length} classes have only ${m} images.`; })()}</div>
      ${srcNote("results/final/final_test_v2.json, results/final/final_test_web_holdout_v1.json")}</div>
    <div class="card"><h3>Per-class recall: sealed test vs web holdout</h3><div class="ctl">${seg("gsort", [["web", "sort by web recall"], ["test", "by test recall"], ["name", "by name"]], "web")}</div><div id="gpc"></div></div>
  </div>
  <div class="card"><h3>Web-holdout errors</h3><p class="small">The weakest web classes mirror training-data contamination: real toggle/slide/DIP <b>switches</b> (the training class is dominated by Nintendo Switch images),
    <b>LEDs</b> predicted as diodes, <b>transformers</b> as inductor coils. Images: Wikimedia Commons, CC/PD licences — hover for author, licence and source (full credits in <code>data/web_holdout_v1/ATTRIBUTION.md</code>).</p>
    <div class="ctl"><label>true class</label><select id="gwc"><option value="">all classes</option>${C.map((c) => `<option>${esc(c)}</option>`).join("")}</select></div><div id="gwe" class="gallery"></div></div>
  <div class="grid g2">
    <div class="card"><h3>All ${num(k.n_files)} original Kaggle files — honestly broken down</h3><div id="gk"></div>
      <p class="small">Each original file is scored against its original folder label. <b>${pct(k.all.accuracy)}</b> overall is <i>not</i> a generalisation estimate:
      ${num(k.by_split.dev.n)} of the files map to rows the head was trained on (${pct(k.by_split.dev.accuracy)}, i.e. training accuracy). The honest number is the
      <b>${k.by_split.test.n} never-seen files</b> (the v2 sealed-test rows plus their duplicates): <b>${pct(k.by_split.test.accuracy)}</b>. The ${k.by_split.excluded.n} files of the conflicting-label groups can be
      right for at most one of their labels (${pct(k.by_split.excluded.accuracy)}).</p>${srcNote("results/kaggle_original/summary.json, results/kaggle_original/per_file.csv")}</div>
    <div class="card"><h3>Never-seen original files: accuracy per class</h3><div id="gkpc"></div></div>
  </div>`;
  function gpc(mode) {
    const r = C.map((c) => ({ c, t: t.per_class[c].recall, w: w.per_class[c].recall, nt: t.per_class[c].support, nw: w.per_class[c].support }));
    if (mode === "web") r.sort((a, b) => a.w - b.w); else if (mode === "test") r.sort((a, b) => a.t - b.t); else r.sort((a, b) => a.c.localeCompare(b.c));
    const W = 600, lw = 150, rh = 22, H = r.length * rh + 30, x = lin(0, 1, lw, W - 30);
    let s = svgOpen(W, H, 'aria-label="per-class recall"') + xAxis(x, 0, H - 26, ticks(0, 1, 5), null, null);
    r.forEach((d, i) => {
      const y = i * rh + 3;
      s += `<text x="${lw - 6}" y="${y + 12}" text-anchor="end" class="lbl">${esc(d.c)}</text>`;
      s += `<rect x="${lw}" y="${y}" width="${x(d.t) - lw}" height="8" fill="var(--accent)" data-tip="${esc(`${d.c}: sealed-test recall ${f3(d.t)} (n = ${d.nt})`)}"/>`;
      s += `<rect x="${lw}" y="${y + 9}" width="${x(d.w) - lw}" height="8" fill="var(--g-invit)" data-tip="${esc(`${d.c}: web-holdout recall ${f3(d.w)} (n = ${d.nw})`)}"/>`;
    });
    $("#gpc", el).innerHTML = s + `</svg><div class="legend"><span><span class="dot" style="background:var(--accent)"></span>sealed test</span><span><span class="dot" style="background:var(--g-invit)"></span>web holdout</span></div>`;
  }
  onSeg(el, "gsort", gpc); gpc("web");
  function gwe() {
    const c = $("#gwc", el).value, { pred, conf } = predsOf(w);
    const idx = [...Array(w.n).keys()].filter((i) => pred[i] !== w.y[i] && (!c || C[w.y[i]] === c));
    $("#gwe", el).innerHTML = idx.map((i) => thumb("w" + w.ids[i], `true <b>${esc(C[w.y[i]])}</b><br>→ ${esc(C[pred[i]])} (${conf[i].toFixed(2)})<br><span class="muted">© ${esc((w.credit[i][0] || "").slice(0, 28))}, ${esc(w.credit[i][1])}</span>`, `${w.ids[i]}` + dsCredit(w, i), conf[i] < D.fit.abstain_threshold ? "uns" : "bad")).join("") || "<p class='muted'>No errors for this class.</p>";
  }
  $("#gwc", el).addEventListener("change", gwe); gwe();
  $("#gk", el).innerHTML = barsH([
    { label: "All files", v: k.all.accuracy, text: `${pct(k.all.accuracy)} (n = ${num(k.all.n)})`, cls: "g-Classical", tip: "includes training rows" },
    { label: "Seen: head fitted on them", v: k.by_split.dev.accuracy, text: `${pct(k.by_split.dev.accuracy)} (n = ${num(k.by_split.dev.n)})`, cls: "g-Self-supervised" },
    { label: "Never seen (v2 test rows)", v: k.by_split.test.accuracy, text: `${pct(k.by_split.test.accuracy)} (n = ${k.by_split.test.n})`, cls: "g-Vision-language" },
    { label: "  … in-scope only", v: k.test_inscope.accuracy, text: `${pct(k.test_inscope.accuracy)} (n = ${k.test_inscope.n})`, cls: "g-Vision-language" },
    { label: "Conflicting-label files", v: k.by_split.excluded.accuracy, text: `${pct(k.by_split.excluded.accuracy)} (n = ${k.by_split.excluded.n})`, cls: "g-From-scratch" },
    { label: "Legacy test, as deployed", v: k.legacy_as_deployed.accuracy, text: `${pct(k.legacy_as_deployed.accuracy)} (n = ${k.legacy_as_deployed.n})`, cls: "g-Classical", tip: "legacy-test files scored by the final model; most were in its training data" },
    { label: "  … unseen rows only", v: k.legacy_unseen.accuracy, text: `${pct(k.legacy_unseen.accuracy)} (n = ${k.legacy_unseen.n})`, cls: "g-ImageNet-CNN" },
    { label: "Legacy-protocol replay", v: rp.all.accuracy, text: `${pct(rp.all.accuracy)} (n = ${rp.all.n}) vs old best ${pct(rp.old_best_for_reference.accuracy)}`, cls: "g-ImageNet-transformer", tip: rp.description },
  ], { w: 620, lw: 175, rw: 205, max: 1, fmt: (v) => Math.round(v * 100) + "%" });
  const kp = Object.entries(k.test_per_class).sort((a, b) => a[1] - b[1]);
  $("#gkpc", el).innerHTML = barsH(kp.map(([c, v]) => ({ label: c, v, text: pct(v), cls: "g-Vision-language" })), { w: 560, lw: 150, rw: 50, max: 1, bh: 14, gap: 4, fmt: (v) => Math.round(v * 100) + "%", ref: k.by_split.test.accuracy, refl: "overall" }) + `<p class="small muted">File-level accuracy on the ${k.by_split.test.n} never-seen files (duplicates count once per file).</p>`;
}
// ==================================================================================================== family diagrams
const DIAG = {};
DIAG.classical = function () {
  const d = dia(940, 250, "HOG + colour histogram + SVM pipeline"); let s = d.s;
  s += bx(10, 90, 96, 70, "image", "img", "→ 128×128");
  s += pathA(d.id, "M106 125 H130 V45 H160"); s += pathA(d.id, "M130 125 V195 H160");
  s += bx(160, 20, 70, 50, "grey", "m"); s += bx(160, 170, 70, 50, "HSV", "m");
  s += ar(d.id, 230, 45, 255, 45); s += ar(d.id, 230, 195, 255, 195);
  // HOG cell grid with orientation glyphs
  const rnd = mulberry(3);
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
    const x = 260 + j * 22, y = 6 + i * 22; s += `<rect x="${x}" y="${y}" width="22" height="22" class="box m" rx="0"/>`;
    for (let k = 0; k < 2; k++) { const a = rnd() * Math.PI, l = 6 + rnd() * 3; s += `<line x1="${x + 11 - l * Math.cos(a)}" y1="${y + 11 - l * Math.sin(a)}" x2="${x + 11 + l * Math.cos(a)}" y2="${y + 11 + l * Math.sin(a)}" stroke="var(--accent)" stroke-width="${1 + k}"/>`; }
  }
  s += tx(354, 30, "HOG: gradient-orientation", "sm", "start") + tx(354, 45, "histograms (9 bins) per", "sm") + tx(354, 60, "16-px cell, 2×2-block norm", "sm") + tx(354, 80, "→ 1764-d", "sm b");
  for (let b = 0; b < 16; b++) { const h = 8 + 34 * Math.abs(Math.sin(b * 1.7 + 0.4)); s += `<rect x="${260 + b * 5.5}" y="${222 - h}" width="4.5" height="${h}" fill="hsl(${b * 22} 60% 55%)"/>`; }
  s += tx(354, 190, "colour histogram", "sm") + tx(354, 205, "8 H × 4 S × 4 V bins", "sm") + tx(354, 222, "→ 128-d", "sm b");
  s += pathA(d.id, "M500 45 H520 V110 H540"); s += pathA(d.id, "M500 205 H520 V140 H540");
  s += bx(540, 95, 100, 60, "concat", "a", "1892-d");
  s += ar(d.id, 640, 125, 665, 125);
  s += bx(665, 95, 105, 60, "Standard-\nScaler", "m");
  s += ar(d.id, 770, 125, 790, 125);
  s += `<rect x="790" y="45" width="140" height="160" rx="8" class="box"/>`;
  const r2 = mulberry(11);
  for (let i = 0; i < 26; i++) { const px = 800 + r2() * 120, py = 55 + r2() * 140, side = (py - 125) - 0.004 * (px - 860) * (px - 860) + 20 * Math.sin((px - 800) / 25) > 0; s += `<circle cx="${px}" cy="${py}" r="3.4" fill="${side ? "var(--g-ssl)" : "var(--accent)"}"/>`; }
  s += `<path d="M795 ${125 - 20 * Math.sin(-0.2)} ${Array.from({ length: 27 }, (_, k) => { const px = 795 + k * 5; return `L${px} ${125 + 0.004 * (px - 860) * (px - 860) - 20 * Math.sin((px - 800) / 25)}`; }).join(" ")}" fill="none" stroke="var(--fg)" stroke-width="2"/>`;
  s += tx(860, 222, "RBF-SVM, C = 10", "sm b", "middle") + tx(860, 238, "max-margin, non-linear boundary", "sm mut", "middle");
  return s + "</svg>";
};
DIAG.scratch = function () {
  const d = dia(960, 280, "scratch CNN feature maps"); let s = d.s;
  const L = [[3, 160, "input"], [32, 80, "stem s2"], [32, 40, "stage 1"], [64, 20, "stage 2"], [128, 10, "stage 3"], [256, 10, "stage 4"]];
  let x = 10;
  L.forEach(([c, sp, lab], i) => {
    const h = 16 + sp * 0.85, w = 6 + Math.log2(c) * 5, y = 140 - h / 2, dx = 12, dy = 9;
    s += `<path d="M${x} ${y} l${dx} ${-dy} h${w} v${h} l${-dx} ${dy} z" class="box ${i ? "a" : "img"}" style="stroke-width:1"/>`;
    s += `<rect x="${x}" y="${y}" width="${w}" height="${h}" class="box ${i ? "a" : "img"}" rx="0"/>`;
    s += tx(x + w / 2 + 6, 244, lab, "sm b", "middle") + tx(x + w / 2 + 6, 260, `${c}×${sp}×${sp}`, "sm mut", "middle");
    const nx = x + w + dx + 52;
    if (i < L.length - 1) s += ar(d.id, x + w + dx + 6, 140, nx - 6, 140);
    x = nx;
  });
  s += ar(d.id, x - 46, 140, x - 6, 140);
  s += bx(x, 120, 80, 40, "GAP", "m", "256"); s += ar(d.id, x + 80, 140, x + 100, 140);
  s += bx(x + 100, 120, 92, 40, "dropout\n+ linear", "g"); s += tx(x + 146, 175, "20 logits", "sm b", "middle");
  s += tx(10, 18, "each stage: [conv3×3 → BN → ReLU] × 2, then max-pool 2 (not in stage 4). Width w = 32 doubles per stage.", "sm");
  s += tx(10, 36, "Stacked 3×3 convs grow the receptive field: two 3×3 = one 5×5 view with fewer weights and an extra non-linearity.", "sm mut");
  return s + "</svg>";
};
DIAG.resnet = function () {
  const d = dia(960, 330, "ResNet bottleneck block and stages"); let s = d.s;
  s += tx(20, 18, "bottleneck block (layer1 of ResNet-50)", "t");
  s += tx(110, 44, "x  (256 ch)", "sm b", "middle"); s += ar(d.id, 110, 50, 110, 64);
  s += bx(30, 64, 160, 36, "1×1 conv 256→64", "a", "BN · ReLU");
  s += ar(d.id, 110, 100, 110, 114); s += bx(30, 114, 160, 36, "3×3 conv 64→64", "a", "BN · ReLU");
  s += ar(d.id, 110, 150, 110, 164); s += bx(30, 164, 160, 36, "1×1 conv 64→256", "a", "BN");
  s += ar(d.id, 110, 200, 110, 222);
  s += `<circle cx="110" cy="236" r="14" class="box o"/>` + tx(110, 241, "+", "t", "middle");
  s += ar(d.id, 110, 250, 110, 270); s += bx(60, 270, 100, 28, "ReLU", "m"); s += tx(110, 318, "y = ReLU(x + F(x))", "sm b", "middle");
  s += pathA(d.id, "M110 54 H230 V236 H126", "g"); s += tx(236, 150, "identity", "sm good") + tx(236, 164, "shortcut", "sm good");
  s += tx(12, 140, "F(x)", "sm b");
  // stage strip
  const st = [["stem", "7×7 s2 + pool", "64×56×56"], ["layer1", "3 blocks", "256×56×56"], ["layer2", "4 blocks", "512×28×28"], ["layer3", "6 blocks", "1024×14×14"], ["layer4", "3 blocks", "2048×7×7"], ["GAP", "", "2048"], ["fc", "", "20"]];
  let x = 320;
  s += tx(320, 100, "ResNet-50 at 224 px (first block of layers 2–4 strides 2, shortcut = 1×1 projection)", "sm");
  st.forEach(([a, b, c], i) => { const w = i < 5 ? 86 : 54; s += bx(x, 120, w, 56, a, i < 5 ? "a" : i === 5 ? "m" : "g", b); s += tx(x + w / 2, 194, c, "sm mut", "middle"); if (i < st.length - 1) s += ar(d.id, x + w, 148, x + w + 12, 148); x += w + 12; });
  s += tx(320, 240, "Why it works: a block that should do nothing only has to push F(x) → 0, and gradients", "sm") + tx(320, 256, "reach early layers directly through the shortcuts — so 50+ layers train well.", "sm");
  return s + "</svg>";
};
DIAG.efficientnet = function () {
  const d = dia(960, 330, "MBConv and Fused-MBConv blocks"); let s = d.s;
  s += tx(10, 18, "MBConv (stages 3–5)", "t");
  const y = 50;
  s += tx(20, y + 34, "x", "b");
  const B = [["1×1 expand", "×k · BN · SiLU", "a"], ["3×3 depthwise", "BN · SiLU", "a"], ["SE", "channel gate", "o"], ["1×1 project", "BN (linear)", "a"]];
  let x = 40;
  B.forEach(([a, b, c], i) => { s += ar(d.id, x, y + 30, x + 18, y + 30); x += 18; s += bx(x, y + 6, 112, 48, a, c, b); x += 112; });
  s += ar(d.id, x, y + 30, x + 18, y + 30); x += 18;
  s += `<circle cx="${x + 13}" cy="${y + 30}" r="13" class="box o"/>` + tx(x + 13, y + 35, "+", "t", "middle");
  s += pathA(d.id, `M28 ${y + 30} V${y - 4} H${x + 13} V${y + 16}`, "g");
  // SE detail
  const sx = 40 + 2 * 130 + 18;
  s += `<path class="ln d" d="M${sx + 56} ${y + 54} V${y + 80}"/>`;
  const se = [["global pool", "C"], ["FC ↓", "C/r"], ["FC ↑", "C"], ["sigmoid", "×"]];
  se.forEach(([a, b], i) => { s += bx(sx - 120 + i * 92, y + 82, 84, 38, a, "m", b); if (i < 3) s += ar(d.id, sx - 36 + i * 92, y + 101, sx - 28 + i * 92, y + 101); });
  s += tx(sx - 120, y + 138, "squeeze-and-excitation: a tiny gate re-weights each channel from the global average", "sm mut");
  s += tx(10, 228, "Fused-MBConv (stages 0–2, high resolution)", "t");
  let x2 = 40;
  s += tx(20, 266, "x", "b");
  [["3×3 conv expand", "×k · BN · SiLU"], ["1×1 project", "BN"]].forEach(([a, b], i) => { s += ar(d.id, x2, 262, x2 + 18, 262); x2 += 18; s += bx(x2, 238, 130, 48, a, "a", b); x2 += 130; });
  s += ar(d.id, x2, 262, x2 + 18, 262); x2 += 18; s += `<circle cx="${x2 + 13}" cy="262" r="13" class="box o"/>` + tx(x2 + 13, 267, "+", "t", "middle");
  s += pathA(d.id, `M28 262 V302 H${x2 + 13} V276`, "g");
  s += tx(400, 262, "one dense 3×3 conv replaces expand + depthwise: slower per FLOP", "sm") + tx(400, 278, "on paper, faster to train on GPUs at high resolution", "sm");
  // compound scaling
  const cx = 760;
  s += tx(cx, 18, "compound scaling", "t");
  s += `<rect x="${cx + 10}" y="70" width="50" height="60" class="box m"/>` + tx(cx + 35, 148, "base (B0)", "sm", "middle");
  s += `<rect x="${cx + 100}" y="40" width="80" height="110" class="box a"/>` + tx(cx + 140, 168, "scaled", "sm", "middle");
  s += ar(d.id, cx + 66, 100, cx + 94, 100);
  s += tx(cx + 186, 60, "↑ depth", "sm b") + tx(cx + 186, 95, "↔ width", "sm b") + tx(cx + 186, 130, "⤢ resolution", "sm b");
  s += tx(cx, 196, "grow all three together by fixed", "sm") + tx(cx, 211, "ratios; V2-S expects 300 px here", "sm");
  return s + "</svg>";
};
DIAG.mobilenet = function () {
  const d = dia(960, 335, "depthwise-separable convolution and inverted residual"); let s = d.s;
  const cube = (x, y, w, h, dep, cls, lab) => `<path d="M${x} ${y} l${dep} ${-dep * 0.6} h${w} v${h} l${-dep} ${dep * 0.6} z" class="box ${cls}" style="stroke-width:1"/><rect x="${x}" y="${y}" width="${w}" height="${h}" class="box ${cls}" rx="0"/>` + (lab ? tx(x + w / 2, y + h + 20, lab, "sm", "middle") : "");
  s += tx(10, 18, "standard 3×3 conv", "t") + tx(10, 34, "mixes space and channels at once", "sm mut");
  s += cube(20, 64, 50, 50, 26, "img", "C_in = 64");
  s += ar(d.id, 100, 89, 140, 89); s += tx(120, 78, "64 filters", "sm", "middle") + tx(120, 112, "3×3×64", "sm", "middle");
  s += cube(150, 64, 50, 50, 26, "a", "C_out = 64");
  s += tx(20, 166, "cost per pixel: 3·3·64·64 = 36,864", "sm b");
  s += tx(340, 18, "depthwise-separable conv", "t") + tx(340, 34, "filter each channel, then mix channels", "sm mut");
  s += cube(340, 64, 50, 50, 26, "img", "C_in");
  s += ar(d.id, 420, 89, 452, 89); s += tx(436, 78, "3×3", "sm", "middle") + tx(436, 112, "per ch.", "sm", "middle");
  s += cube(460, 64, 50, 50, 26, "o", "depthwise");
  s += ar(d.id, 540, 89, 572, 89); s += tx(556, 78, "1×1", "sm", "middle") + tx(556, 112, "×64", "sm", "middle");
  s += cube(580, 64, 50, 50, 26, "a", "pointwise");
  s += tx(340, 166, "cost: 3·3·64 + 64·64 = 576 + 4,096 = 4,672  →  ≈ 7.9× cheaper", "sm b");
  s += tx(10, 206, "inverted residual (MobileNetV2/V3 block): thin → expand → depthwise → linear project → thin", "t");
  const widths = [[24, "x (24 ch)", "img"], [96, "1×1 expand ×4", "a"], [96, "k×k depthwise", "o"], [96, "SE (some blocks)", "m"], [24, "1×1 linear project", "a"]];
  let x = 20;
  widths.forEach(([wd, lab, cls], i) => { const h = wd * 0.9; s += `<rect x="${x}" y="${272 - h / 2}" width="34" height="${h}" class="box ${cls}"/>` + tx(x + 17, 272 + h / 2 + 14, lab, "sm", "middle"); if (i < 4) s += ar(d.id, x + 34, 272, x + 120, 272); x += 155; });
  s += pathA(d.id, `M37 ${272 - 13} V222 H${20 + 4 * 155 + 17} V${272 - 13}`, "g");
  s += tx(720, 262, "skip joins the thin ends; the last", "sm") + tx(720, 277, "projection has no activation (linear)", "sm") + tx(720, 292, "hard-swish = x·ReLU6(x+3)/6 in deep layers", "sm mut");
  return s + "</svg>";
};
DIAG.convnext = function () {
  const d = dia(960, 360, "ResNet bottleneck vs ConvNeXt block"); let s = d.s;
  const col = (x0, title, rows) => {
    let t = tx(x0 + 120, 20, title, "t", "middle"), y = 36;
    rows.forEach(([lab, w, cls], i) => { const ww = 120 + w * 0.3; t += bx(x0 + 120 - ww / 2, y, ww, 28, lab, cls); if (i < rows.length - 1) t += ar(d.id, x0 + 120, y + 28, x0 + 120, y + 38); y += 38; });
    t += `<circle cx="${x0 + 120}" cy="${y + 8}" r="12" class="box"/>` + tx(x0 + 120, y + 13, "+", "t", "middle");
    t += ar(d.id, x0 + 120, y - 10, x0 + 120, y - 4);
    t += pathA(d.id, `M${x0 + 120} 30 H${x0 + 262} V${y + 8} H${x0 + 134}`, "g");
    return t;
  };
  s += col(10, "ResNet-50 bottleneck (stage 1)", [["1×1 conv 256→64", 64, "a"], ["BN · ReLU", 64, "m"], ["3×3 conv 64→64", 64, "a"], ["BN · ReLU", 64, "m"], ["1×1 conv 64→256", 256, "a"], ["BN", 256, "m"]]);
  s += tx(130, 300, "wide → narrow → wide; ReLU after the sum", "sm mut", "middle");
  s += col(330, "ConvNeXt-T block (stage 1)", [["7×7 depthwise, 96", 96, "o"], ["LayerNorm", 96, "m"], ["1×1 conv 96→384", 384, "a"], ["GELU", 384, "m"], ["1×1 conv 384→96", 96, "a"], ["layer scale γ · drop path", 96, "m"]]);
  s += tx(450, 300, "narrow → wide → narrow (inverted bottleneck)", "sm mut", "middle");
  const notes = ["large 7×7 depthwise kernel (spatial mixing,", "  like per-channel attention windows)", "LayerNorm instead of BatchNorm, one per block", "a single GELU, fewer normalisations", "4×4 stride-4 'patchify' stem", "separate 2×2 stride-2 downsampling layers", "stage ratio 3:3:9:3 (Swin-like)", "no attention anywhere"];
  s += tx(660, 36, "what changed from ResNet", "t");
  notes.forEach((n, i) => { s += tx(660, 60 + i * 22, (n.startsWith(" ") ? "  " : "• ") + n.trim(), "sm"); });
  return s + "</svg>";
};
DIAG.vit = function () {
  const d = dia(960, 330, "Vision Transformer"); let s = d.s;
  s += `<rect x="20" y="70" width="140" height="140" class="box img"/>`;
  for (let i = 1; i < 7; i++) s += `<line x1="${20 + i * 20}" y1="70" x2="${20 + i * 20}" y2="210" stroke="var(--fg2)" stroke-width="0.6"/><line x1="20" y1="${70 + i * 20}" x2="160" y2="${70 + i * 20}" stroke="var(--fg2)" stroke-width="0.6"/>`;
  s += tx(90, 230, "224 px → 14×14 = 196", "sm", "middle") + tx(90, 245, "patches of 16×16 (7×7 drawn)", "sm mut", "middle");
  s += ar(d.id, 165, 140, 200, 140); s += tx(182, 128, "linear", "sm", "middle");
  // token column
  for (let i = 0; i < 7; i++) { const y = 62 + i * 24; s += `<rect x="205" y="${y}" width="44" height="18" rx="3" class="${i ? "tok" : "tokc"}"/>` + tx(227, y + 13, i ? (i === 6 ? "…" : "p" + i) : "CLS", "sm", "middle"); }
  s += tx(227, 248, "197 × 768", "sm b", "middle") + tx(227, 50, "+ pos. emb.", "sm", "middle");
  s += ar(d.id, 252, 140, 290, 140);
  // encoder block
  s += `<rect x="292" y="32" width="300" height="230" rx="10" class="box m"/>` + tx(442, 52, "Transformer encoder block  × 12", "t", "middle");
  s += bx(312, 70, 100, 30, "LayerNorm", ""); s += ar(d.id, 412, 85, 432, 85); s += bx(432, 66, 140, 38, "multi-head", "a", "self-attention · 12 heads");
  s += `<circle cx="442" cy="132" r="11" class="box o"/>` + tx(442, 137, "+", "t", "middle");
  s += pathA(d.id, "M502 104 V132 H455");
  s += pathA(d.id, "M302 85 V132 H429", "g");
  s += bx(312, 165, 100, 30, "LayerNorm", ""); s += ar(d.id, 412, 180, 432, 180); s += bx(432, 160, 140, 40, "MLP", "a", "768 → 3072 → 768, GELU");
  s += `<circle cx="442" cy="232" r="11" class="box o"/>` + tx(442, 237, "+", "t", "middle");
  s += pathA(d.id, "M502 200 V232 H455"); s += pathA(d.id, "M442 143 V152 H302 V232 H429", "g");
  s += `<path class="ln" d="M362 100 V115"/>`;
  s += ar(d.id, 592, 140, 630, 140);
  s += bx(632, 110, 110, 60, "LayerNorm", "m", "take [CLS] → 768");
  s += ar(d.id, 742, 140, 772, 140); s += bx(772, 115, 100, 50, "Linear", "g", "768 → 20");
  s += tx(632, 210, "attention(Q,K,V) = softmax(QKᵀ/√d)·V", "sm b") + tx(632, 228, "every token sees every token from layer 1;", "sm") + tx(632, 244, "cost grows with tokens² (196² here)", "sm mut");
  s += tx(292, 290, "pre-norm residual block: t ← t + MHSA(LN(t)); t ← t + MLP(LN(t))", "sm");
  return s + "</svg>";
};
DIAG.swin = function () {
  const d = dia(960, 345, "Swin shifted windows and hierarchy"); let s = d.s;
  const cell = 19, grid = (x0, y0, shift, title) => {
    let t = tx(x0 + 4 * cell, y0 - 12, title, "t", "middle");
    for (let i = 0; i < 8; i++) for (let j = 0; j < 8; j++) {
      const wi = Math.floor(((i + shift) % 8) / 4), wj = Math.floor(((j + shift) % 8) / 4), hue = [205, 150, 35, 280][wi * 2 + wj];
      const region = shift ? (Math.floor((i < 8 - shift ? i + shift : i + shift - 8) / 4) * 2 + Math.floor((j < 8 - shift ? j + shift : j + shift - 8) / 4)) : wi * 2 + wj;
      t += `<rect x="${x0 + j * cell}" y="${y0 + i * cell}" width="${cell - 1}" height="${cell - 1}" fill="hsl(${[205, 150, 35, 280][region]} 55% 70%)" opacity="0.8"/>`;
    }
    if (!shift) { t += `<path d="M${x0 + 4 * cell} ${y0} v${8 * cell} M${x0} ${y0 + 4 * cell} h${8 * cell}" stroke="var(--fg)" stroke-width="2.5"/>`; }
    else { t += `<path d="M${x0 + 2 * cell} ${y0} v${8 * cell} M${x0 + 6 * cell} ${y0} v${8 * cell} M${x0} ${y0 + 2 * cell} h${8 * cell} M${x0} ${y0 + 6 * cell} h${8 * cell}" stroke="var(--fg)" stroke-width="2.5"/>`; }
    t += `<rect x="${x0}" y="${y0}" width="${8 * cell}" height="${8 * cell}" fill="none" stroke="var(--fg)" stroke-width="1.5"/>`;
    return t;
  };
  s += grid(20, 46, 0, "block l: W-MSA");
  s += grid(230, 46, 2, "block l+1: SW-MSA");
  s += tx(20, 225, "illustration: 8×8 tokens, 4×4 windows; Swin-T uses 7×7-token", "sm") + tx(20, 241, "windows, shifted by 3. Attention only inside each thick-bordered", "sm") + tx(20, 257, "window; the shift lets information cross the previous borders.", "sm") + tx(20, 273, "(colour = window before the shift)", "sm mut");
  const P = [[56, 96], [28, 192], [14, 384], [7, 768]];
  let x = 470;
  s += tx(470, 24, "hierarchy (patch merging halves H, W and doubles C)", "t");
  P.forEach(([hw, c], i) => { const h = 30 + hw * 2.6; s += `<rect x="${x}" y="${170 - h / 2}" width="${40 + c / 30}" height="${h}" class="box a"/>` + tx(x + (40 + c / 30) / 2, 170 + h / 2 + 18, `${hw}×${hw}×${c}`, "sm", "middle") + tx(x + (40 + c / 30) / 2, 170 + h / 2 + 33, `stage ${i + 1}`, "sm mut", "middle"); const nx = x + 40 + c / 30 + 34; if (i < 3) s += ar(d.id, x + 40 + c / 30 + 3, 170, nx - 3, 170); x = nx; });
  s += tx(470, 335, "strides 4, 8, 16, 32 like a CNN; 2 / 2 / 6 / 2 blocks; 3 / 6 / 12 / 24 heads", "sm mut");
  return s + "</svg>";
};
DIAG.dinov2 = function () {
  const d = dia(960, 340, "DINO self-distillation"); let s = d.s;
  s += `<rect x="20" y="120" width="90" height="90" class="box img"/>` + tx(65, 230, "one image", "sm", "middle");
  s += `<rect x="150" y="40" width="56" height="56" class="box img"/><rect x="214" y="40" width="56" height="56" class="box img"/>` + tx(210, 32, "2 global crops", "sm", "middle");
  for (let i = 0; i < 4; i++) s += `<rect x="${150 + i * 32}" y="240" width="26" height="26" class="box img"/>`;
  s += tx(210, 285, "several local crops", "sm", "middle");
  s += pathA(d.id, "M110 150 L146 80"); s += pathA(d.id, "M110 185 L146 245");
  s += bx(340, 30, 160, 70, "teacher ViT", "p", "global crops only");
  s += bx(340, 210, 160, 70, "student ViT", "a", "all crops (+ masked patches)");
  s += pathA(d.id, "M270 68 H336"); s += pathA(d.id, "M270 68 H300 V230 H336"); s += pathA(d.id, "M278 253 H336");
  s += pathA(d.id, "M420 210 V104", "d"); s += tx(428, 160, "EMA of student weights", "sm b") + tx(428, 175, "(no gradient to teacher)", "sm mut");
  s += ar(d.id, 500, 65, 540, 65); s += bx(540, 40, 170, 50, "centre + sharpen", "m", "softmax over K prototypes");
  s += ar(d.id, 500, 245, 540, 245); s += bx(540, 220, 170, 50, "softmax", "m", "over K prototypes");
  s += pathA(d.id, "M710 65 H760 V140"); s += pathA(d.id, "M710 245 H760 V176");
  s += bx(720, 140, 220, 36, "cross-entropy(teacher, student)", "o");
  s += tx(720, 196, "match predictions across DIFFERENT crops", "sm") + tx(720, 212, "→ features encode what the object is", "sm");
  s += tx(720, 250, "iBOT: student sees masked patches and", "sm") + tx(720, 266, "predicts the teacher's patch tokens", "sm") + tx(720, 282, "→ good dense (per-patch) features", "sm");
  s += tx(20, 325, "DINOv2: LVD-142M curated images, ViT-g/14 teacher with DINO + iBOT (+ KoLeo); ViT-S/B/L used here are distilled from it.", "sm mut");
  return s + "</svg>";
};
DIAG.clip = function () {
  const d = dia(960, 380, "CLIP dual encoder and contrastive matrix"); let s = d.s;
  const N = 4, cs = 38, mx = 360, my = 100;
  for (let i = 0; i < N; i++) s += `<rect x="20" y="${my + i * cs + 4}" width="34" height="30" class="box img"/>`;
  s += tx(37, my - 10, "images", "sm b", "middle");
  s += pathA(d.id, `M56 ${my + 75} H68`);
  s += bx(70, my + 40, 110, 70, "image\nencoder", "a", "ViT");
  s += ar(d.id, 180, my + 75, 210, my + 75);
  for (let i = 0; i < N; i++) s += `<rect x="215" y="${my + i * cs + 8}" width="60" height="22" rx="3" class="tok"/>` + tx(245, my + i * cs + 23, `I${i + 1}`, "sm", "middle");
  s += ar(d.id, 280, my + 75, 350, my + 75);
  const caps = ["a red LED", "a relay module", "a 10k resistor", "a breadboard"];
  caps.forEach((c, j) => { s += `<text transform="translate(${mx + j * cs + 22},${my - 8}) rotate(-35)" class="sm">"${c}"</text>`; });
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) s += `<rect x="${mx + j * cs}" y="${my + i * cs}" width="${cs - 2}" height="${cs - 2}" class="${i === j ? "cellp" : "celln"}" data-tip="${esc(`I${i + 1}·T${j + 1}${i === j ? " — matching pair (target)" : " — mismatched pair"}`)}"/>` + tx(mx + j * cs + cs / 2 - 1, my + i * cs + cs / 2 + 3, i === j ? "✓" : "", "sm", "middle");
  for (let j = 0; j < N; j++) s += `<rect x="${mx + j * cs + 4}" y="${my + N * cs + 12}" width="30" height="20" rx="3" class="tokc"/>` + tx(mx + j * cs + 19, my + N * cs + 26, `T${j + 1}`, "sm", "middle");
  s += bx(mx - 10, my + N * cs + 62, N * cs + 20, 40, "text encoder", "p", "captions of the same batch");
  s += ar(d.id, mx + N * cs / 2, my + N * cs + 62, mx + N * cs / 2, my + N * cs + 36);
  s += tx(mx + N * cs + 14, my + 30, "softmax over each ROW:", "sm b") + tx(mx + N * cs + 14, my + 46, "image picks its caption", "sm");
  s += tx(mx + N * cs + 14, my + 80, "softmax over each COLUMN:", "sm b") + tx(mx + N * cs + 14, my + 96, "caption picks its image", "sm");
  s += tx(mx + N * cs + 14, my + 130, "loss = ½ (CE rows + CE cols),", "sm") + tx(mx + N * cs + 14, my + 146, "logits = I·T / τ (learned τ)", "sm");
  s += `<rect x="740" y="60" width="210" height="200" rx="10" class="box m"/>` + tx(845, 82, "zero-shot classifier", "t", "middle");
  s += tx(752, 108, "\"a photo of a {class}.\" × 6 templates", "sm") + tx(752, 126, "→ text encoder → average", "sm") + tx(752, 144, "→ 20 class embeddings", "sm");
  s += tx(752, 172, "image embedding · each class", "sm") + tx(752, 190, "(cosine) → argmax", "sm") + tx(752, 218, "no training images needed", "sm b") + tx(752, 242, "homonyms: switch, fuse, relay…", "sm bad");
  return s + "</svg>";
};
DIAG.siglip = function () {
  const d = dia(960, 340, "SigLIP sigmoid loss and MAP pooling"); let s = d.s;
  const N = 4, cs = 40, mx = 30, my = 60;
  s += tx(mx, 30, "sigmoid loss: every pair is its own yes/no", "t");
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) s += `<rect x="${mx + 30 + j * cs}" y="${my + i * cs}" width="${cs - 2}" height="${cs - 2}" class="${i === j ? "cellp" : "celln"}" data-tip="${esc(`pair (I${i + 1}, T${j + 1}): z = ${i === j ? "+1" : "−1"}; loss −log σ(z·(t·I·T + b))`)}"/>` + tx(mx + 30 + j * cs + cs / 2 - 1, my + i * cs + cs / 2 + 4, i === j ? "+1" : "−1", "sm", "middle");
  for (let i = 0; i < N; i++) s += tx(mx + 18, my + i * cs + 24, `I${i + 1}`, "sm", "middle");
  for (let j = 0; j < N; j++) s += tx(mx + 30 + j * cs + cs / 2, my - 6, `T${j + 1}`, "sm", "middle");
  s += tx(mx, my + N * cs + 22, "loss_ij = −log σ( z_ij · (t · I_i·T_j + b) )", "sm b");
  s += tx(mx, my + N * cs + 40, "no softmax over the batch → no all-gather,", "sm") + tx(mx, my + N * cs + 56, "works at moderate batch size; bias b offsets", "sm") + tx(mx, my + N * cs + 72, "N positives vs N²−N negatives", "sm");
  // MAP head
  const ox = 360;
  s += tx(ox, 30, "MAP head (multihead attention pooling)", "t");
  for (let i = 0; i < 9; i++) s += `<rect x="${ox + i * 34}" y="200" width="28" height="20" rx="3" class="tok"/>` + tx(ox + i * 34 + 14, 214, i === 8 ? "…" : "", "sm", "middle");
  s += tx(ox + 150, 240, "729 patch tokens × 1152 (So400m @ 384; no [CLS])", "sm", "middle");
  s += `<rect x="${ox + 125}" y="70" width="40" height="22" rx="3" class="tokc"/>` + tx(ox + 145, 85, "q", "sm b", "middle") + tx(ox + 175, 85, "learned query", "sm");
  for (let i = 0; i < 9; i++) s += `<line x1="${ox + 145}" y1="92" x2="${ox + i * 34 + 14}" y2="198" stroke="var(--warn)" stroke-width="${0.6 + 1.6 * Math.abs(Math.sin(i * 1.3))}" opacity="0.8"/>`;
  s += pathA(d.id, `M${ox + 145} 70 V50 H${ox + 300}`); s += bx(ox + 302, 30, 100, 40, "MLP", "m", "→ 1152-d");
  s += tx(ox, 270, "attention weights decide which patches form the image", "sm") + tx(ox, 286, "feature; in this repo the MAP block is part of the backbone", "sm");
  s += `<rect x="790" y="40" width="160" height="220" rx="10" class="box m"/>` + tx(870, 62, "So400m/14 @ 384", "t", "middle");
  ["patch 14 → 27×27", "= 729 tokens", "width 1152", "27 blocks", "16 heads × 72", "MLP 4304", "(mlp_ratio 3.7362)", "shape-optimised ViT"].forEach((l, i) => { s += tx(802, 88 + i * 20, l, "sm"); });
  return s + "</svg>";
};

// ==================================================================================================== 8. FAMILIES
function renderFamilies(el) {
  const fams = D.families;
  const bestOf = (f) => D.bench.filter((b) => b.family === f.id && b.schedule !== "old (bug)").reduce((a, b) => (!a || b.f1 > a.f1 ? b : a), null);
  el.innerHTML = `<h2>Model families — architecture, pretraining, knobs and results</h2>
  <p class="lead">Eleven families from a hand-crafted pipeline to vision-language foundation models. Each page has the core idea, an
  architecture diagram, a step-by-step walk-through with the tensor shapes of the exact model used here, a characteristics table, the
  hyperparameters with their repo config keys, the measured results, failure modes, commands to try and references — plus the full
  explainer from <code>docs/models/</code>.</p>
  ${legendGroups()}
  <div class="famgrid" role="group" aria-label="families">${fams.map((f) => { const b = bestOf(f); return `<button type="button" class="famcard" data-f="${f.id}" aria-pressed="false" style="border-top-color:var(--${{ Classical: "g-classical", "From scratch": "g-scratch", "ImageNet CNN": "g-incnn", "ImageNet transformer": "g-invit", "Self-supervised": "g-ssl", "Vision-language": "g-vl" }[f.group]})">
    <div class="n">${esc(f.name)}</div><div class="tiny muted">${esc(f.group)} · ${esc(f.year)}</div><div class="tg">${f.tagline}</div>
    <div class="small">best here: <b>${b ? f3(b.f1) : "—"}</b> macro-F1 <span class="muted">(${b ? esc(b.regime) : ""})</span></div></button>`; }).join("")}</div>
  <div id="famdetail" style="margin-top:18px"></div>`;
  const det = $("#famdetail", el);
  function show(id, scroll) {
    const i = fams.findIndex((f) => f.id === id), f = fams[i >= 0 ? i : 0];
    $$(".famcard", el).forEach((b) => b.setAttribute("aria-pressed", b.dataset.f === f.id));
    const rows = D.bench.filter((b) => b.family === f.id).sort((a, b) => b.f1 - a.f1);
    const ftRows = rows.filter((b) => b.params_M != null);
    const prRows = rows.filter((b) => b.feature_dim);
    const paired = D.paired.filter((p) => rows.some((r) => r.backbone === p.backbone));
    const best = rows.find((b) => b.schedule !== "old (bug)");
    const sec = (t) => f.sections.find((x) => x.title.toLowerCase().includes(t));
    const failure = sec("failure"), tryit = sec("try"), refs = sec("reference");
    const others = f.sections.filter((x) => x !== failure && x !== tryit && x !== refs);
    const charRows = [];
    if (ftRows.length) charRows.push(["Parameters (20-class head)", Array.from(new Set(ftRows.map((b) => `${b.params_M.toFixed(1)}M (${esc(b.label)})`))).join("<br>")]);
    if (prRows.length) charRows.push(["Frozen feature dimension", Array.from(new Set(prRows.map((b) => `${b.feature_dim} (${esc(pr(b))})`))).join("<br>")]);
    Object.entries(f.chars).forEach(([k, v]) => charRows.push([esc(k), v]));
    function pr(b) { return b.label.replace(" + flip TTA", ""); }
    const prev = fams[(i - 1 + fams.length) % fams.length], next = fams[(i + 1) % fams.length];
    det.innerHTML = `
    <div class="famnav"><button class="btn" type="button" data-nav="${prev.id}">← ${esc(prev.name)}</button><button class="btn" type="button" data-nav="${next.id}">${esc(next.name)} →</button></div>
    <div class="card"><h2 style="margin-bottom:2px">${esc(f.name)} <span class="badge">${esc(f.group)}</span> <span class="badge">${esc(f.year)}</span></h2>
      <p class="small muted">${f.meta}</p>
      <div class="grid g21"><div><h3>Core idea and history</h3>${f.idea.map((p) => `<p>${p}</p>`).join("")}
        <h4>Key papers</h4><ul class="small">${f.papers.map((p) => `<li>${p}</li>`).join("")}</ul>
        <div class="callout"><b>Lesson from this project.</b> ${f.lesson}</div></div>
      <div><h3>Characteristics</h3><table>${charRows.map(([k, v]) => `<tr><th style="position:static;width:38%">${k}</th><td>${v}</td></tr>`).join("")}</table>
        <p class="tiny muted">FLOPs only where the original paper reports them (at the paper's resolution); not measured here. Parameters and feature dimensions from the run records.</p></div></div></div>
    <div class="card"><h3>Architecture</h3>${(DIAG[f.diagram] || (() => ""))()}${f.id === "siglip" ? "<h4 style='margin-top:14px'>…and the final model built on it</h4>" + finalDiagram() : ""}
      <p class="diacap">Schematic drawn for this explorer; numbers in the labels follow the walk-through below.</p></div>
    <div class="card"><h3>How a picture flows through it</h3><p class="small muted">${f.flow_title}</p>
      <div class="ctl"><button class="btn" type="button" id="fprev">◀ previous step</button><button class="btn primary" type="button" id="fnext">next step ▶</button><span class="small muted" id="fstep"></span></div>
      <table class="flow"><thead><tr><th>#</th><th>stage</th><th>operation</th><th>output shape</th></tr></thead><tbody>${f.flow.map((r, k) => `<tr data-k="${k}"><td class="num">${k + 1}</td><td><b>${esc(r[0])}</b></td><td>${esc(r[1])}</td><td class="shape">${esc(r[2])}</td></tr>`).join("")}</tbody></table></div>
    <div class="card"><h3>Measured in this project (grouped 5-fold CV)</h3>
      ${rows.length ? `<div class="tablewrap"><table><tr><th>Method</th><th>Regime</th><th>Schedule</th><th class="num">Folds</th><th class="num">Macro-F1</th><th class="num">Accuracy</th><th class="num">In-scope acc.</th><th class="num">Trainable</th><th class="num">Min/fold</th><th class="num">VRAM GB</th></tr>
      ${rows.map((b) => `<tr class="${b === best ? "hl" : ""}"><td>${esc(b.label)}</td><td>${esc(b.regime)}</td><td>${esc(b.schedule)}</td><td class="num">${b.folds}</td><td class="num">${f4(b.f1)} ± ${f4(b.f1_sd)}</td><td class="num">${f4(b.acc)}</td><td class="num">${f4(b.inscope_acc)}</td><td class="num">${b.trainable_M == null ? "—" : b.trainable_M < 1 ? (b.trainable_M * 1000).toFixed(1) + "k" : b.trainable_M.toFixed(1) + "M"}</td><td class="num">${b.train_min ?? "—"}</td><td class="num">${b.vram_gb ?? "—"}</td></tr>`).join("")}</table></div>` : ""}
      <p><b>Adaptation regimes.</b> ${f.regimes} <b>Best here:</b> ${best ? `${esc(best.label)} (${esc(best.regime)}), macro-F1 ${f3(best.f1)} ± ${f3(best.f1_sd)} — rank ${D.bench.filter((b) => b.schedule !== "old (bug)").sort((a, b) => b.f1 - a.f1).indexOf(best) + 1} of ${D.bench.filter((b) => b.schedule !== "old (bug)").length} methods (old-schedule rows excluded).` : "—"}</p>
      ${paired.map((p) => `<p class="small">Paired on folds ${p.folds.join(", ")}: fine-tune − probe = <b>${pts(p.mean_delta, 2)} pts</b>, fine-tune wins ${p.wins} folds (${p.delta.map((x) => pts(x, 2)).join(", ")}).</p>`).join("")}
      ${rows.some((b) => b.history) ? `<div class="ctl"><label>training curves</label><select id="fcurve">${rows.filter((b) => b.history).map((b, k) => `<option value="${k}">${esc(b.label)} — ${esc(b.schedule)}</option>`).join("")}</select></div><div id="fcurvechart"></div>` : ""}
      ${srcNote("results/benchmark.csv + per-fold run records")}</div>
    <div class="card"><h3>Hyperparameters that matter (repo config keys)</h3>
      ${f.configs.length ? f.configs.map((c) => `<h4><code>${esc(c.file)}</code> — effective recipe after merging <code>configs/base.yaml</code></h4><div class="tablewrap"><table><tr><th>key</th><th>value</th><th>set in</th><th>what it does / what changing it does</th></tr>${c.hp.map(([k, v, own, ex]) => `<tr><td><code>${esc(k)}</code></td><td><code>${esc(v)}</code></td><td class="small">${own ? "<b>family file</b>" : "base.yaml"}</td><td class="small">${ex ? esc(ex).replace(/`([^`]+)`/g, "<code>$1</code>") : ""}</td></tr>`).join("")}</table></div>
        <details class="doc"><summary>raw <code>${esc(c.file)}</code></summary><pre class="code"><code>${esc(c.raw)}</code></pre></details>`).join("")
        : `<table><tr><th>knob</th><th>meaning</th></tr>${f.hp.map(([k, v]) => `<tr><td><code>${esc(k)}</code></td><td>${v}</td></tr>`).join("")}</table>`}
      <p class="tiny muted">Override any key from the command line: <code>uv run python -m compclass.train --config configs/&lt;x&gt;.yaml --fold 0 --set train.lr=5.0e-5</code>.</p></div>
    <div class="grid g2">
      <div class="card"><h3>Typical failure modes</h3><div class="docbody">${failure ? failure.html : "<p class='muted'>—</p>"}</div></div>
      <div class="card"><h3>Try it</h3><div class="docbody">${tryit ? tryit.html : "<p class='muted'>—</p>"}</div></div></div>
    <div class="card"><h3>Full explainer (from <code>docs/models/${esc(f.doc)}.md</code>)</h3>
      ${others.map((x, k) => `<details class="doc" ${k === 0 ? "open" : ""}><summary>${x.title}</summary><div class="docbody">${x.html}</div></details>`).join("")}
      ${refs ? `<h4>References</h4><div class="docbody">${refs.html}</div>` : ""}
      <p class="tiny muted">The doc's "Results on data v2" placeholder section is replaced by the measured table above.</p></div>`;
    // flow stepping
    let k = -1;
    const trs = $$(".flow tbody tr", det);
    const step = (dlt) => { k = Math.max(0, Math.min(trs.length - 1, k + dlt)); trs.forEach((t, j) => t.classList.toggle("cur", j === k)); $("#fstep", det).textContent = `step ${k + 1} of ${trs.length}: ${f.flow[k][0]} → ${f.flow[k][2]}`; };
    $("#fnext", det).addEventListener("click", () => step(1)); $("#fprev", det).addEventListener("click", () => step(-1));
    const cs = $("#fcurve", det);
    if (cs) { const hr = rows.filter((b) => b.history); const drawC = () => { $("#fcurvechart", det).innerHTML = curveSVG(hr[+cs.value], 640); }; cs.addEventListener("change", drawC); drawC(); }
    if (scroll) det.scrollIntoView({ block: "start" });
  }
  el.addEventListener("click", (e) => {
    const c = e.target.closest("[data-f]"), n = e.target.closest("[data-nav]");
    const id = c ? c.dataset.f : n ? n.dataset.nav : null;
    if (id) { show(id, true); try { history.replaceState(null, "", "#families/" + id); } catch (er) { /* sandbox */ } }
  });
  el.__sub = (s, scroll) => show(s, scroll);
  show("siglip", false);
}

// ==================================================================================================== 9. GLOSSARY + REPRODUCE
function renderGlossary(el) {
  el.innerHTML = `<h2>Glossary</h2><div class="ctl"><label for="gq">search</label><input type="search" id="gq" placeholder="e.g. calibration, LLRD, leakage" size="40"><span id="gn" class="small muted"></span></div><dl class="glossary card" id="gl"></dl>`;
  const draw = () => {
    const q = $("#gq", el).value.trim().toLowerCase();
    const hits = D.glossary.filter(([t, d]) => !q || t.toLowerCase().includes(q) || d.toLowerCase().replace(/<[^>]+>/g, "").includes(q));
    const hl = (s) => (q ? s.replace(new RegExp("(" + q.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + ")(?![^<]*>)", "gi"), '<span class="searchhit">$1</span>') : s);
    $("#gl", el).innerHTML = hits.map(([t, d]) => `<dt>${hl(esc(t))}</dt><dd>${hl(d)}</dd>`).join("") || "<dd class='muted'>No match.</dd>";
    $("#gn", el).textContent = `${hits.length} of ${D.glossary.length} terms`;
  };
  $("#gq", el).addEventListener("input", draw); draw();
}
function renderReproduce(el) {
  el.innerHTML = `<h2>Reproduce</h2><p class="lead">Commands per phase, from <code>README.md</code>, <code>docs/DEMO.md</code> and the script docstrings. GPU phases ran on one
  RTX 4060 Ti (8 GB). The sealed test and the web holdout have each been read once; re-running those commands would be a second look.</p>
  ${D.reproduce.map(([t, note, cmds], i) => `<div class="card"><h3>${esc(t)}</h3><p class="small muted">${esc(note)}</p><div class="copyrow"><pre class="code" id="cmd${i}"><code>${esc(cmds.join("\n"))}</code></pre><button class="btn" type="button" data-copy="cmd${i}">copy</button></div></div>`).join("")}
  <div class="card"><h3>About this page</h3><p class="small">Built by <code>scripts/build_explorer.py</code> from repository files only (git ${esc(D.meta.head || "?")}, data v2 <code>${esc(D.meta.manifest)}</code>,
  web holdout <code>${esc(D.meta.web_fingerprint)}</code>). The build asserts that the sealed-test and web-holdout metrics recomputed from the embedded probabilities equal the result JSONs, that
  <code>results/benchmark.csv</code> equals the mean of the per-fold records, and that the headline numbers in the text match their source files.</p></div>`;
  el.addEventListener("click", (e) => {
    const b = e.target.closest("[data-copy]"); if (!b) return;
    const pre = $("#" + b.dataset.copy, el), txt = pre.textContent;
    const done = () => { b.textContent = "copied"; setTimeout(() => (b.textContent = "copy"), 1200); };
    try { navigator.clipboard.writeText(txt).then(done, () => sel(pre)); } catch (er) { sel(pre); }
    function sel(p) { const r = document.createRange(); r.selectNodeContents(p); const s = getSelection(); s.removeAllRanges(); s.addRange(r); b.textContent = "selected — Ctrl+C"; }
  });
}

// ==================================================================================================== init
function route() { const h = (location.hash || "#overview").slice(1).split("&")[0]; const [t, s] = h.split("/"); activate(t || "overview", s || null, false); }
window.addEventListener("hashchange", route);
route();
window.__explorerReady = true;
})();

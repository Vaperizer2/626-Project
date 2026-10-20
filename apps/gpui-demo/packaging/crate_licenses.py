"""Print the Rust crates linked into gpui-demo with version, licence and source, from `cargo metadata`.

Reads `cargo metadata --format-version 1 --features ... --filter-platform <triple>` on stdin and walks the resolve
graph from the root package over normal (non-dev) and build dependencies. Used by scripts/package_linux.sh.
"""

import json
import sys

meta = json.load(sys.stdin)
pkgs = {p["id"]: p for p in meta["packages"]}
nodes = {n["id"]: n for n in meta["resolve"]["nodes"]}
root = meta["resolve"]["root"]
seen, stack = set(), [root]
while stack:
    pid = stack.pop()
    if pid in seen:
        continue
    seen.add(pid)
    for dep in nodes[pid]["deps"]:
        kinds = {k["kind"] for k in dep["dep_kinds"]}
        if kinds - {"dev"}:
            stack.append(dep["pkg"])
seen.discard(root)
rows = sorted((pkgs[i]["name"], pkgs[i]["version"], pkgs[i].get("license") or "(see crate)",
               pkgs[i].get("repository") or "") for i in seen)
by_lic = {}
for _, _, lic, _ in rows:
    by_lic[lic] = by_lic.get(lic, 0) + 1
print("Third-party Rust crates compiled into gpui-demo (from Cargo.lock; licence = the crate's SPDX field)")
print("Full texts: Apache-2.0.txt, MIT-template.txt (each crate's copyright line is in its source package).")
print("MPL-2.0 crates are used unmodified; their source is at the repository listed. Where a crate offers a choice")
print("(\"A OR B\"), this distribution uses it under the first permissive licence listed (MIT or Apache-2.0).")
print("Summary: " + "; ".join(f"{k}: {v}" for k, v in sorted(by_lic.items(), key=lambda kv: -kv[1])))
print()
for name, ver, lic, repo in rows:
    print(f"{name} {ver}\t{lic}\t{repo}")

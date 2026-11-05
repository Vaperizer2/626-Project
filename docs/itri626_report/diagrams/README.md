# Excalidraw report diagrams

Created through the actual `mcp__excalidraw__create_view` tool on 5 November 2026 after its `read_me` reference.
Final MCP checkpoints: workflow `57eddc119c2a4e8ea2`; protocols `7a59f2a56a914dc29d`.

For each diagram:

- `*.elements.json`: original MCP element array, including its camera update.
- `*.excalidraw`: editable standard Excalidraw scene.
- `*.svg`: official Excalidraw 0.18.0 vector export of the same elements.
- `*.pdf`: SVG converted by `rsvg-convert` for LaTeX inclusion.

The official Excalidraw library's `convertToExcalidrawElements`, `exportToSvg` (20 px padding, white background)
and `serializeAsJSON` produced the exports in a headless Chromium browser. This is an export of the MCP scene,
not a Mermaid or independently redrawn substitute. Text uses Excalidraw's sans-serif font family (2).
The workflows contain no external images or photo attribution dependencies.

To revise, open the `.excalidraw` file in Excalidraw, export SVG with a white background and update the PDF:

```sh
rsvg-convert -f pdf -o workflow.pdf workflow.svg
rsvg-convert -f pdf -o protocols.pdf protocols.svg
```

If editing through the MCP, update the input array and re-export the standard scene too, so all formats agree.
The report source includes the PDFs; editing the scene alone does not rebuild those PDFs automatically.

# Electronic component classification comparative

Self-contained Jupyter notebooks. **All dependency installs go through [`uv`](https://docs.astral.sh/uv/).**

| Notebook | Role |
|---|---|
| [`notebooks/01_primary_comparative.ipynb`](notebooks/01_primary_comparative.ipynb) | Fair comparative: HOG→SVM, SmallCNN, ResNet18 |
| [`notebooks/02_playground.ipynb`](notebooks/02_playground.ipynb) | Extra families / sizes / params |

---

## Setup: initialize the environment with `uv`

### 1. Install `uv` (once per machine)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# or: pacman / brew / pipx — see https://docs.astral.sh/uv/getting-started/installation/
uv --version
```

### 2. Create / sync the project env (recommended)

From the repo root:

```bash
cd /path/to/626-Project
uv sync --all-groups
```

This creates `.venv/` from `pyproject.toml` + `uv.lock` (torch, sklearn, jupyter, kaggle, timm, …).

### 3. Register a Jupyter kernel (once)

```bash
uv run python -m ipykernel install --user --name=compclass --display-name="Python (compclass)"
```

### 4. Open a notebook

**Cursor / VS Code:** open the `.ipynb` → select kernel **Python (compclass)** or `.venv/bin/python`.

**Terminal:**

```bash
uv run jupyter notebook notebooks/01_primary_comparative.ipynb
# or
uv run jupyter lab notebooks/01_primary_comparative.ipynb
```

Skip the notebook “Install” cell if you already ran `uv sync` — packages are already in `.venv`.

### Notebook-only (no repo checkout)

If someone only has the `.ipynb`:

```bash
mkdir component-lab && cd component-lab
uv venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
uv pip install ipykernel notebook
uv run python -m ipykernel install --user --name=compclass --display-name="Python (compclass)"
```

Then open the notebook with that kernel and run the **Install** cell — it calls `uv pip install --python …` into the active env.

---

## Data

- **Kaggle API:** `uv run kaggle auth login`, or `KAGGLE_API_TOKEN`, or `~/.kaggle/access_token`
- **Manual:** drop the [dataset](https://www.kaggle.com/datasets/mdfaisalahmed025/electronic-component-recognizer) `.zip` or ImageFolder under `data/raw/` (created next to the notebook cwd)

## Outputs

- `outputs/figures/` — confusion matrices (also embedded in the saved notebook)
- `outputs/weights/` — model weights
- `outputs/metrics/primary_metrics.json` — numbers for a later LaTeX write-up
- `data/splits/split_manifest.json` — frozen 70/15/15 split

## Adding packages later

Always use `uv`, not `pip`:

```bash
uv add some-package              # project (updates pyproject + lock)
uv add --dev timm                # playground extras
uv pip install --python .venv/bin/python some-package   # ad-hoc into venv
```

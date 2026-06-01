# Setup

## Python environment

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Runtime dependencies (`requirements.txt`): `numpy`, `pandas`, `scipy`,
`scikit-learn`, `lightfm`, `pyyaml`, `tqdm`, `matplotlib`, `requests`,
`python-dotenv`, `pyarrow`.

> `lightfm` builds native code and needs a C/C++ toolchain (Xcode CLI on macOS,
> `build-essential` on Linux, MSVC build tools on Windows). Install that first if the
> wheel build fails.

There is no linter or formal test runner configured. The `tests/` directory holds two
runnable smoke scripts (below), not a `pytest` suite.

## Dataset (gitignored — must be downloaded)

The pipeline reads the Kaggle BoardGameGeek bulk dump
[`threnjen/board-games-database-from-boardgamegeek`](https://www.kaggle.com/datasets/threnjen/board-games-database-from-boardgamegeek)
(~21.9K games, ~19M ratings). It is **not** in the repo — `data/` is gitignored.

Place the unzipped CSVs directly under `data/raw/` so that `games.csv`,
`user_ratings.csv`, and `mechanics.csv` sit at `data/raw/<file>.csv`. Two ways:

- **Manual:** download the `.zip` from the dataset page (free Kaggle account required),
  unzip into `data/raw/`.
- **CLI:** `pip install kaggle`, create an API token (`kaggle.json` at
  `%USERPROFILE%\.kaggle\kaggle.json`), then
  `kaggle datasets download -d threnjen/board-games-database-from-boardgamegeek -p data/raw --unzip`.

Full walkthrough and license note: [`../plans/m1_plan.md`](../plans/m1_plan.md).

## LLM server (context annotation only)

Tier-3 context annotation (`scripts/annotate_context.py`) queries an **Ollama** server
over its REST API. The host/port are secrets and live in `.env`; the model and sampling
options are non-secret and live in `configs/data.yaml` under `llm:`.

```bash
cp .env.example .env        # then set OLLAMA_HOST to the real host/IP
```

`.env` keys (see [configuration.md](configuration.md#env)):

```
OLLAMA_HOST=board-llm.tailXXXX.ts.net   # Tailscale MagicDNS name or 100.x IP
OLLAMA_PORT=11434
```

`.env` is gitignored — never commit it.

## Smoke tests

Both require a reachable Ollama server (they make live requests).

```bash
# Connectivity + one sample JSON-constrained prompt:
python tests/test_llm.py      --config configs/data.yaml

# Full annotation pipeline on 3 in-memory games (no games.parquet needed):
python tests/test_annotate.py --config configs/data.yaml
```

`test_annotate.py` asserts label correctness on Codenames / Twilight Imperium /
Pandemic (e.g. Pandemic must come back `coop_vs_competitive == "coop"`, and Twilight
Imperium's omitted age fields must resolve to `pd.NA`).

# Local development (DeepSeek API / Ollama)

Iterate on `env.py` and `benchanything.json` on your machine before `mesocosm env submit`.

Two options for local model inference:

| Provider | When to use | Auth |
|----------|-------------|------|
| **DeepSeek API** (default) | You have a DeepSeek API key | `DEEPSEEK_API_KEY` env var |
| **Ollama** (fallback) | No API key, want fully local inference | None (runs locally) |

## One-time setup

1. Install the CLI: `pip install swecc-mesocosm`
   This covers `mesocosm run local`, `bench_common` for `adapter.py`, and the HTTP stack (`fastapi`, `uvicorn`). You do **not** need `pip install -r requirements.txt` for the default scaffold — that file is only for extra packages your env imports (see comments in `requirements.txt`). The platform installs it when you `env submit`.

### Option A — DeepSeek API (default)

2. Get a DeepSeek API key from [platform.deepseek.com](https://platform.deepseek.com) and set it:
   ```bash
   export DEEPSEEK_API_KEY=sk-...
   ```
   (Add to `~/.bashrc` or use a `.env` file.)

### Option B — Ollama (fallback, no API key needed)

2. Install Ollama and pull a model:
   ```bash
   ollama pull llama3.2
   ```
3. Ensure Ollama is running (`ollama serve` — the desktop app usually does this).

## Dev loop

```bash
export MESOCOSM_LOCAL=1   # optional: bench-api :8010, adapter :8765 defaults
mesocosm doctor --local   # verify adapter (8765) before run local
```

**Terminal 1 — env server**

```bash
python adapter.py
# → http://localhost:8765/health
```

**Terminal 2 — bench episodes**

With DeepSeek API (default):
```bash
mesocosm run local
# same as: mesocosm run local --model deepseek/deepseek-reasoner
```

With Ollama (fallback):
```bash
mesocosm run local --model ollama/llama3.2
```

Uses `benchanything.json` for the binding vow and scoring. Does **not** register the domain or create platform runs.

## Flags

| Flag | Default | Purpose |
|------|---------|---------|
| `--model` | `deepseek/deepseek-reasoner` | Provider/model via LiteLLM (`deepseek/*`, `ollama/*`, `openai/*`, etc.) |
| `--episodes` | `5` | Number of episodes |
| `--env-url` | `http://localhost:8765` | Adapter URL if you changed the port |
| `--manifest` | `benchanything.json` | Alternate manifest path |
| `--system-prompt` | — | Extra instruction for the agent |

> **Tip:** Set `--model ollama/llama3.2` to switch to local Ollama inference (no API key required).

## Ship to Mesocosm

When local runs look good:

```bash
mesocosm auth login
mesocosm env submit --name "My env" --github-url https://github.com/you/your-repo
# submit clones the repo and registers a draft domain from benchanything.json — no separate register step
mesocosm env list   # note domain_id when status is ready
mesocosm run create --domain DOMAIN_ID --vow-version 1.0.0 --model gemini/gemini-3.1-flash-lite ...
```

Platform runs use cloud models on SWECC infrastructure; local inference (DeepSeek API or Ollama) is only for your machine.

**Non-interactive auth:** `mesocosm auth login` prompts for credentials. In CI, set `SWECC_BENCH_TOKEN` or use `mesocosm auth guest`.

**Legacy:** repos that use `domain.py` with `DOMAIN_CONFIG` (not created by `mesocosm init`) can still run `mesocosm register path/to/domain.py [--auto-id] [--publish]` to POST the domain manually.

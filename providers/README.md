# providers/

Model-backend plugins. Each `*.py` here declares a module-level `PROVIDER` dict
(and an optional helpers-only `setup(lc, cfg)` hook) that teaches lean-coder to talk
to one backend - a hosted API, an OAuth subscription, or a local server. A provider
is **not** a lean-tool: it wires in through a native registry, never ships to a
`/connect` remote, and exposes no model-facing tool.

## What's here

- **`ollama.py`** - the bundled default (`provider = "ollama"`); local/self-hosted
  models. Also the worked reference: HTTP client, priority-pool host failover,
  per-host last-used model, num_ctx auto-detect + OOM cap, and its own `/ollama`
  command. Read this first when writing your own.
- **`anthropic_api.py`, `gemini.py`, `groq.py`, `openai.py`, `openrouter.py`,
  `anthropic_plan.py`** - bundled hosted providers, disabled until you enable + key
  them (`/provider enable <name>`, then `/provider login <name>`).

## Writing one

Full contract, every `PROVIDER` key, the `setup(lc, cfg)` injection model, and how to
register a command are in **[`../PROVIDER_API.md`](../PROVIDER_API.md)**. Start from
the annotated template at
**[`../examples/providers/example.py`](../examples/providers/example.py)**.

Drop your file here (or in `~/.config/leancoder/providers/`), then
`/provider enable <name>`. Config autosaves - no manual save step.

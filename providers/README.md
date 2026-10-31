# providers/

Model-backend plugins. Each `*.py` here declares a module-level `PROVIDER` dict (and
an optional helpers-only `setup(lc, cfg)`) for one backend: a hosted API, an OAuth
subscription, or a local server. See [PROVIDER_API.md](../doc/PROVIDER_API.md) for how a
provider differs from a lean-tool.

## Files

- `ollama.py` - the default (`provider = "ollama"`) for local or self-hosted models.
  Also the reference implementation: HTTP client, host failover, per-host last-used
  model, num_ctx detection and OOM cap, and its own `/ollama` command.
- Hosted providers (`gemini.py`, `groq.py`, `openai.py`, `openrouter.py`, `kilo.py`,
  `ovh.py`, and others) are off until enabled and keyed: `/provider enable <name>`,
  then `/provider login <name>`. `kilo` needs no key (anonymous lane, 200 requests/hour
  per IP); `ovh` needs none either (OVHcloud, EU-hosted, 2 requests/minute per model per
  IP).
- `llamacpp.py`, `mlx.py` - local servers (llama.cpp, Apple MLX).

## Writing one

The contract is in [PROVIDER_API.md](../doc/PROVIDER_API.md). Start from
[`../examples/providers/example.py`](../examples/providers/example.py).

Put your file here (or in `~/.config/leancoder/providers/`), then
`/provider enable <name>`.

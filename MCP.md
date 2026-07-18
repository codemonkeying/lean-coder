# Using MCP servers with lean-coder

lean-coder is a **generic [MCP](https://modelcontextprotocol.io) client**, built into
the core. Point it at any MCP server and that server's tools join the model's tool
surface, namespaced `mcp__<server>__<tool>`. Nothing ships enabled - **zero servers by
default**, so MCP costs no context and adds no tools until you add one.

This is the counterpart to a [lean-tool](LEAN_TOOLS.md): a lean-tool is a `.py` file you
*write*; an MCP server is an external process/endpoint you *connect to*. Use a lean-tool
for something small and local; use MCP to plug in an existing server from the wider
ecosystem.

---

## Quick start

```
/mcp add fs npx -y @modelcontextprotocol/server-filesystem /some/dir   # a stdio server
/mcp add gw https://mcp-gateway.example.com/mcp/handbook/mcp           # an HTTP server
/mcp                       # enable/disable menu (per server, live)
/mcp list                  # configured servers + connection state
/mcp reconnect [name]      # (re)connect all enabled, or just one
/mcp remove <name>         # forget a server
/mcp clean [name]          # drop stale entries (enabled but never defined)
```

`/mcp add <name> <spec>` infers the transport from the spec: a `http(s)://` URL ->
**HTTP**; anything else -> a **stdio** command line. The server is saved to your config
and enabled immediately; its tools appear on the next turn.

**Guided auth.** If an HTTP server you add (or reconnect) answers `401`/`403`, lean-coder
does not just dump a red error - it offers an inline auth walkthrough (the same shape as a
provider that needs a login): pick **OAuth 2.1** or **static bearer**, answer a couple of
prompts, and it wires the auth up and reconnects. For OAuth against a gateway that supports
dynamic client registration this is fully hands-off - see
[Guided OAuth (auto-registration)](#guided-oauth-auto-registration) below. No TTY (piped /
headless)? It prints the exact config block to add instead.

Every subcommand also answers `/mcp ?` with its full help.

---

## The two transports

Both are **stdlib-only** (no `mcp` SDK, no extra pip deps). HTTP uses `curl` if present.

### stdio

A spawned subprocess speaking JSON-RPC 2.0 over its stdin/stdout (newline-delimited) -
the same shape Claude Desktop uses. Configured with `command` + `args` + `env`:

```toml
[mcp_servers.fs]
transport = "stdio"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "/some/dir"]
# env = { SOME_TOKEN = "..." }   # merged over the current environment
```

### HTTP (streamable)

The streamable-HTTP MCP flow: `initialize` -> capture the `Mcp-Session-Id` header ->
`notifications/initialized` -> `tools/list` / `tools/call`. Responses may be **SSE**
(`event: message` / `data: {…}`) or plain JSON - both are handled. Session-less servers
(no `Mcp-Session-Id`) work too.

```toml
[mcp_servers.gw]
transport = "http"
url = "https://mcp-gateway.example.com/mcp/handbook/mcp"
```

---

## Authentication (HTTP only)

Auth reduces to a single `Authorization: Bearer <token>` header. Two ways to produce it:

**Prefer OAuth 2.1 where the gateway offers it.** A static bearer token is a long-lived
secret: if it leaks it works until someone rotates it by hand. OAuth 2.1 client
credentials issue a short-lived JWT (typically ~1h) that lean-coder fetches, caches, and
refetches automatically, so a leaked token expires on its own and the client secret never
rides on the wire per request. Use a static bearer only when the server has no OAuth
endpoint.

### Static bearer token

```toml
[mcp_servers.gw]
transport = "http"
url = "https://…/mcp/handbook/mcp"
auth = { type = "bearer", token_env = "GW_KEY" }   # or: token = "literal-token"
```

Prefer `token_env` (read from the environment at call time) over a literal `token` in
the file, so no secret is written to `config.toml`.

### OAuth 2.1 (client credentials)

For a gateway that issues short-lived JWTs. lean-coder fetches a token from `token_url`,
caches it, and **refetches automatically on a 401**:

```toml
[mcp_servers.gw]
transport = "http"
url = "https://…/mcp/handbook/mcp"
auth = { type = "oauth", token_url = "https://…/oauth/token", client_id = "my-client", client_secret_env = "GW_SECRET", scope = "mcp:access" }
```

`client_secret_env` (env lookup) is preferred; `client_secret` (literal) is accepted.
`scope` is optional. Both `bearer`-static and `oauth` end as the same one Bearer header,
so a gateway that accepts either kind Just Works.

### Guided OAuth (auto-registration)

You rarely need to hand-write the `oauth` block above. When an HTTP server returns
`401`/`403`, the guided walkthrough can set the whole thing up for you, including
**dynamic client registration (DCR, RFC 7591)** so you never have to mint a client by
hand. Choosing OAuth in the prompt does:

1. **Discovery** - fetches the gateway's `/.well-known/oauth-protected-resource`
   (RFC 9728) then `/.well-known/oauth-authorization-server` (RFC 8414) to find the
   `token_endpoint` and `registration_endpoint`.
2. **Registration** - if a `registration_endpoint` exists, it POSTs a client-credentials
   registration (passing a registration key as a Bearer header if the endpoint is guarded
   - it asks you for one, blank if the endpoint is open) and gets back a
   `client_id` + `client_secret`.
3. **Storage** - writes those to a **mode-0600 sidecar** at
   `~/.config/leancoder/mcp_auth/<server>.json`, and sets `config.toml` to just
   `auth = { type = "oauth" }`. **The secret never touches `config.toml`.**
4. **Connect** - fetches the first token, caches it (with its expiry) back into the
   sidecar, and reconnects.

If the gateway has no DCR endpoint, the walkthrough falls back to asking for the
`token_url` / `client_id` / secret manually; you can put the secret in an env var
(`client_secret_env`) or let lean-coder store it in the same 0600 sidecar.

**Where secrets live (the rule).** Anything *you supply* (a static bearer token, a
pre-existing OAuth client secret) belongs in an **environment variable** referenced by
`token_env` / `client_secret_env` - never written to disk by lean-coder. Anything
lean-coder *mints itself* (a DCR client secret, cached access tokens) goes in the
**0600 sidecar** under the config dir, the same convention the `anthropic_plan` provider
uses for `auth.json`. Either way, `config.toml` stays free of secrets.

The sidecar (`~/.config/leancoder/mcp_auth/<server>.json`) holds
`{client_id, client_secret, token_endpoint, registration_endpoint, scope}` plus the
cached `{access_token, expires_at}`; delete the file to force a fresh registration.

---

## How MCP tools behave

- **Naming.** A server `gw` exposing `search_handbook` surfaces as
  `mcp__gw__search_handbook`. The namespace keeps MCP tools from colliding with core
  tools, lean-tools, or each other.
- **Driver-only.** MCP tools always run on the **driver** (the machine running
  lean-coder), never on a `/connect`-ed remote executor. The subprocess/endpoint lives
  where lean-coder does.
- **Leash tier.** MCP tools may have side effects, so they ride the **`rwe`** tier (like
  a non-`safe` lean-tool) - available only at the `rwe` leash, and they confirm before
  running unless approval is armed (`/approve session` or `auto`).
- **Startup.** Enabled servers connect at launch. A server that fails to connect just
  contributes no tools; its error is shown by `/mcp list` (and you can `/mcp reconnect`
  it once fixed). A dead server never blocks the loop.
- **Results.** A tool result's `structuredContent` is preferred, else text content
  blocks are joined; an `isError` result is prefixed `[tool error]`.

---

## Config reference

Two keys in `config.toml`, both empty by default:

```toml
# Servers you've added (name -> spec). Written by /mcp add; edit by hand for auth/env.
[mcp_servers.<name>]
transport = "stdio" | "http"
# stdio:
command = "…"
args = ["…"]
env = { KEY = "val" }
# http:
url = "https://…"
auth = { type = "bearer", token_env = "…" }   # or the oauth form above

# Which servers are live on the tool surface (per-server gate).
mcp_enabled = ["gw", "fs"]
```

`/mcp` (menu), `/mcp add`, `/mcp remove`, and the enable/disable menu all persist these
for you - you only edit by hand to set `auth`/`env`, which the `add` shortcut doesn't
prompt for.

---

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `/mcp list` shows `error: …` | Server unreachable / bad command / auth rejected. Fix, then `/mcp reconnect <name>`. |
| `0 tools` but connected | Some proxy servers list nothing until after `initialize` - lean-coder always handshakes first, so this usually means the upstream itself is empty or its own auth failed. |
| HTTP server needs `curl` | The HTTP transport shells out to `curl`; install it, or use a stdio server. |
| Tool never offered to the model | Raise the leash to `rwe` (`/leash rwe`) - MCP tools ride that tier. |
| Secret ended up in `config.toml` | Use `token_env` / `client_secret_env` instead of the literal forms. |

# MCP servers

lean-coder has a stdlib-only [MCP](https://modelcontextprotocol.io) client. A server's
tools appear as `mcp__<server>__<tool>`. None are configured by default.

## Commands

```
/mcp add fs npx -y @modelcontextprotocol/server-filesystem /some/dir   # stdio
/mcp add gw https://mcp-gateway.example.com/mcp/handbook/mcp           # http
/mcp                       # enable/disable menu
/mcp list                  # servers + connection state
/mcp reconnect [name]
/mcp remove <name>
/mcp clean [name]          # drop stale entries
```

A `http(s)://` spec means HTTP; anything else is a stdio command. On a `401`/`403`,
`/mcp add` and `/mcp reconnect` offer a guided OAuth or bearer setup (without a TTY
they print the config block instead). `/mcp ?` shows full help.

## Config

```toml
[mcp_servers.fs]
transport = "stdio"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "/some/dir"]
env = { SOME_TOKEN = "..." }        # merged over the environment

[mcp_servers.gw]
transport = "http"                  # needs curl on PATH
url = "https://mcp-gateway.example.com/mcp/handbook/mcp"
auth = { type = "bearer", token_env = "GW_KEY" }   # or token = "literal"
# auth = { type = "oauth", token_url = "https://.../oauth/token", client_id = "my-client",
#          client_secret_env = "GW_SECRET", scope = "mcp:access" }

mcp_enabled = ["gw", "fs"]
```

- OAuth 2.1 client credentials: token fetched, cached, refetched on 401. `scope` optional.
- Guided OAuth uses discovery (RFC 9728, RFC 8414) and dynamic client registration
  (RFC 7591) when available, and stores minted credentials and cached tokens in
  `~/.config/leancoder/mcp_auth/<server>.json` (mode 0600), with `auth = { type = "oauth" }`
  in `config.toml`. Delete the file to force re-registration.
- Use `token_env` / `client_secret_env` so no secret is written to `config.toml`.

## Behaviour

- MCP tools run on the driver, never on a `/connect` remote.
- They are `rwe` tier and confirm unless approval is armed.
- Enabled servers connect at launch; a failed one contributes no tools and shows its
  error in `/mcp list`.
- Results use `structuredContent` if present, else joined text; `isError` results are
  prefixed `[tool error]`.

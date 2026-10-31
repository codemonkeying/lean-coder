# Workers and task boards

## `dispatch_worker`

Runs a sub-task in a background `lean_coder --agent-run` process with its own context.

`dispatch_worker(task=..., [model], [cwd], [leash], [host], [iterations], [prompt],
[context], [taskboard], [tools], [from_session])`, with `action` (default `dispatch`):

| action | does |
|---|---|
| `dispatch` | start a worker from `task` |
| `models` | list usable models |
| `status` | all workers, or one `pid` |
| `result` | full result (`pid`) |
| `transcript` | worker's tool trail (`pid`, `page`); needs `worker_checkpoint` |
| `cancel` | kill and discard |
| `pause` | save as a session and stop |
| `resume` | restart with a fresh budget; `text` adds a steer |
| `inject` | send `text` to a running worker |

- `tools=` restricts the worker's tools (names the driver has); `update_plan`, `note`,
  `request_compact` are always kept.
- `prompt=` adds a role prompt: `code_review`, `online_research`, or
  `prompts/system/workers/<name>.txt`.
- `leash` defaults to `r`, capped at the driver's leash.
- Inference uses the driver's provider and account (`worker_inherit`); tools go
  through the driver's executor (remote under `/connect`).
- The tool is `driver_only`.

`/worker` lists workers; `/worker <pid>` prints a result; `/worker status|cancel|inject`.

### Settings

| setting | env var | default |
|---|---|---|
| `worker_max_concurrent` | `LEANCODER_WORKER_MAX_CONCURRENT` | 10 |
| `worker_idle_timeout` | `LEANCODER_WORKER_IDLE_TIMEOUT` | 1800 |
| `worker_default_iterations` | `LEANCODER_WORKER_DEFAULT_ITER` | 50 |
| `worker_max_iterations` | `LEANCODER_WORKER_MAX_ITER` | 200 (0 = unlimited) |
| `worker_max_depth` | `LEANCODER_WORKER_MAX_DEPTH` | 2 |
| `worker_max_children` | `LEANCODER_WORKER_MAX_CHILDREN` | 2 |
| `worker_checkpoint` | | false |

A worker that exhausts its budget saves a checkpoint and reports `INCOMPLETE`; continue
with `action='resume'`.

## Task boards (`board`)

A task graph stored at `workers/taskboards/<name>.json` under the config dir.

- Lifecycle: open -> assigned -> done | failed | cancelled. A task is ready when all
  its `deps` are done.
- Driver-only: `create`, `add`, `assign`, `cancel`, `participant` registration.
  Workers: `done`, `fail`, `list`, `reconcile`.
- `reconcile` returns `result_ref`s in dependency order.
- Dispatching with `taskboard=<name>` gives the worker `board`.

`/board`, `/board <name>`, `/board <name> rm <id> [...]`, `/board delete <name>`.

## Wake on background finish

`wake_on_bg_finish = true` (default): a background task or worker finishing while the
prompt is idle starts a turn with the result. Otherwise the notice rides the next
typed turn. Not active with piped stdin.

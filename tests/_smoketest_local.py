"""Local-only companion to _smoketest.py: tests for personal provider plugins
(anthropic, etc.) that ship with canon but NOT the public repo. _smoketest.py
exec()s this file if it is present, sharing its globals (check, lc, Path, FIX);
a public checkout without it simply skips these checks. Never committed to public.
"""
# 25. anthropic_plan.py provider (was backend.py): _estimate_cost, usage, formatters  # sweep-ok
import sys as _sys
import io as _io
be = lc._load_lean_tool(Path(lc.__file__).parent / "providers" / "anthropic_plan.py")  # sweep-ok

# cost estimates
check("backend._estimate_cost: haiku",
      abs(be._estimate_cost("claude-haiku-4-5-20251001", 1_000_000, 0) - 1.0) < 0.001)  # sweep-ok
check("backend._estimate_cost: sonnet",
      abs(be._estimate_cost("claude-sonnet-4-6", 0, 1_000_000) - 15.0) < 0.001)  # sweep-ok
check("backend._estimate_cost: opus-4-6",
      abs(be._estimate_cost("claude-opus-4-6", 1_000_000, 1_000_000) - 30.0) < 0.001)  # sweep-ok
check("backend._estimate_cost: opus (old)",
      abs(be._estimate_cost("claude-opus-4-20250514", 1_000_000, 0) - 15.0) < 0.001)  # sweep-ok
check("backend._estimate_cost: unknown -> None",
      be._estimate_cost("unknown-model", 100, 100) is None)

# cache_boundary -> _cvt_messages propagates the flag; both vendor clients turn it into
# the 4th cache breakpoint and strip the internal flag before send.
_api = lc._load_lean_tool(Path(lc.__file__).parent / "providers" / "anthropic_api.py")  # sweep-ok
for _mod, _nm in ((be, "plan"), (_api, "api")):
    _sys_s, _apimsgs = _mod._cvt_messages([
        {"role": "system", "content": "S"},
        {"role": "user", "content": "summary", "cache_boundary": True},
        {"role": "user", "content": "recent-1"},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "recent-2"},
    ])
    check(f"[{_nm}] _cvt_messages propagates cache_boundary onto the summary msg",
          _apimsgs[0].get("cache_boundary") is True)
    # simulate the client's breakpoint pass: mark [-2] (rolling) + the boundary, strip flag
    _cm = list(_apimsgs)
    if len(_cm) >= 2:
        _cm[-2] = _mod._mark_cache(_cm[-2])
    _bnd = max((k for k, mm in enumerate(_cm) if mm.get("cache_boundary")), default=None)
    if _bnd is not None and _bnd != len(_cm) - 2:
        _cm[_bnd] = _mod._mark_cache(_cm[_bnd])
    _cm = [{k: v for k, v in mm.items() if k != "cache_boundary"} for mm in _cm]
    def _has_cc(msg):
        c = msg.get("content")
        return (isinstance(c, list) and c and "cache_control" in c[-1])
    check(f"[{_nm}] boundary msg gets a cache_control breakpoint", _has_cc(_cm[0]))
    check(f"[{_nm}] cache_boundary flag stripped before send",
          not any("cache_boundary" in mm for mm in _cm))
    _bps = sum(1 for mm in _cm if _has_cc(mm))
    check(f"[{_nm}] exactly 2 message breakpoints (rolling + boundary)", _bps == 2)
    # non-string tool content (a loaded/synthesised history) must not 400 the request:
    # _tool_result_content coerces None/number to a string.
    check(f"[{_nm}] _tool_result_content coerces None to a string",
          _mod._tool_result_content({"content": None}) == "")
    check(f"[{_nm}] _tool_result_content coerces a number to a string",
          _mod._tool_result_content({"content": 42}) == "42")
lc._lean_tool_commands.clear(); lc._providers.clear()

# usage() provider callback: feeds meters to core renderer
lc._lean_tool_commands.clear()
lc._providers.clear()
_cfg = lc.Config(cwd=FIX)
be.setup(vars(lc), _cfg)
agent = lc.Agent(_cfg)
_usage_fn = lc.get_provider("anthropic_plan")["usage"]  # sweep-ok
class _FakeApiClientUsage:
    _last_usage = {
        "five_hour": {"utilization": 12.0, "resets_at": "2026-06-26T18:00:00Z"},
        "seven_day": {"utilization": 61.0, "resets_at": "2026-06-27T22:00:00Z"},
        "limits": [],
    }
agent.client = _FakeApiClientUsage()
_meters = _usage_fn(agent, _cfg)
check("backend: usage() returns meters dict",   _meters is not None and "meters" in _meters)
check("backend: usage() 5h meter label",        _meters["meters"][0]["label"] == "5h")
check("backend: usage() 5h meter pct",          _meters["meters"][0]["pct"] == 12.0)
check("backend: usage() wk meter label",        _meters["meters"][1]["label"] == "wk")
check("backend: usage() wk meter pct",          _meters["meters"][1]["pct"] == 61.0)
check("backend: usage() 5h resets_at present",  _meters["meters"][0]["resets_at"] == "2026-06-26T18:00:00Z")
class _FakeApiClientNoUsage:
    _last_usage = {}
agent.client = _FakeApiClientNoUsage()
check("backend: usage() returns None when no usage data", _usage_fn(agent, _cfg) is None)
lc._lean_tool_commands.clear()
lc._providers.clear()

# _fmt_reset and _fmt_usage_banner
check("backend._fmt_reset: formats ISO UTC to local short string",
      ":" in be._fmt_reset("2026-06-27T22:00:00Z") and be._fmt_reset("2026-06-27T22:00:00Z") != "")
check("backend._fmt_reset: handles empty input gracefully",
      be._fmt_reset("") == "")
check("backend._fmt_reset: handles bad input without raising",
      isinstance(be._fmt_reset("notadate"), str))

_sample_usage_throttled = {
    "five_hour": {"utilization": 5.0, "resets_at": "2026-06-26T15:00:00Z"},
    "seven_day": {"utilization": 56.0, "resets_at": "2026-06-27T22:00:00Z"},
    "limits": [{"kind": "weekly_all", "percent": 56, "is_active": True}],
}
_banner_throttled = be._fmt_usage_banner(_sample_usage_throttled)
check("backend._fmt_usage_banner: shows 5h pct", "5%" in _banner_throttled)
check("backend._fmt_usage_banner: shows 7d pct", "56%" in _banner_throttled)
check("backend._fmt_usage_banner: shows active limit warning",
      "weekly_all" in _banner_throttled and "throttled" in _banner_throttled)

_sample_usage_ok = {
    "five_hour": {"utilization": 10.0, "resets_at": "2026-06-26T15:00:00Z"},
    "seven_day": {"utilization": 20.0, "resets_at": "2026-06-27T22:00:00Z"},
    "limits": [],
}
_banner_ok = be._fmt_usage_banner(_sample_usage_ok)
check("backend._fmt_usage_banner: no warning when no active limits", "[!" not in _banner_ok)
check("backend._fmt_usage_banner: returns empty string on empty input",
      be._fmt_usage_banner({}) == "")

# _sn short number formatter
check("backend._sn: 500 -> '500'",    be._sn(500)       == "500")
check("backend._sn: 1000 -> '1k'",    be._sn(1000)      == "1k")
check("backend._sn: 1250 -> '1.2k'",  be._sn(1250)      == "1.2k")
check("backend._sn: 200000 -> '200k'", be._sn(200000)   == "200k")
check("backend._sn: 1000000 -> '1m'", be._sn(1_000_000) == "1m")

# _fmt_5h_rst and _fmt_7d_rst
import datetime as _dt
_5h_iso = "2026-06-26T18:00:00Z"
_7d_iso = "2026-07-02T22:00:00Z"
_5h_exp = _dt.datetime.fromisoformat(_5h_iso.replace("Z","+00:00")).astimezone().strftime("%H:%M")
_7d_exp = _dt.datetime.fromisoformat(_7d_iso.replace("Z","+00:00")).astimezone().strftime("(%a)%d%b")
check("backend._fmt_5h_rst: HH:MM local",  be._fmt_5h_rst(_5h_iso) == _5h_exp)
check("backend._fmt_5h_rst: empty -> ''",  be._fmt_5h_rst("") == "")
check("backend._fmt_7d_rst: ddMMM local",  be._fmt_7d_rst(_7d_iso) == _7d_exp)
check("backend._fmt_7d_rst: empty -> ''",  be._fmt_7d_rst("") == "")


# _bar progress bar
check("backend._bar: 0% -> all light",  be._bar(0,  10) == "░" * 10)
check("backend._bar: 100% -> all dark", be._bar(100, 10) == "█" * 10)
check("backend._bar: 50% -> half",      be._bar(50, 10) == "█" * 5 + "░" * 5)
check("backend._bar: default width 20", len(be._bar(73)) == 20)

# _fmt_remaining
import datetime as _dt2
_far = (_dt2.datetime.now(_dt2.timezone.utc) + _dt2.timedelta(hours=5, minutes=30)).isoformat().replace("+00:00", "Z")
_near = (_dt2.datetime.now(_dt2.timezone.utc) + _dt2.timedelta(minutes=45)).isoformat().replace("+00:00", "Z")
_past = "2020-01-01T00:00:00Z"
check("backend._fmt_remaining: 5.5h future starts with '5h'", be._fmt_remaining(_far).startswith("5h"))
check("backend._fmt_remaining: 45m future -> Xm (no hours)",  be._fmt_remaining(_near).endswith("m") and "h" not in be._fmt_remaining(_near))
check("backend._fmt_remaining: past -> 'now'",                be._fmt_remaining(_past) == "now")
check("backend._fmt_remaining: empty -> ''",                  be._fmt_remaining("") == "")


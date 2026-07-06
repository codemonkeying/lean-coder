"""/provision - set up lean_coder on another box, end to end.

A wizard built on the setup() hook. Enable it with /tools, then run
`/provision [user@host]`. It walks you through:

  1. install lean_coder on the box  (shells to install.sh --remote)
  2. pick local lean-tools to copy over  (multi-select)
  3. pick addresses to seed in its config  (your [machines] + this box's LAN
     IPs, so a same-LAN box can reach your Ollama with no manual setup)

It is confirm-gated, reuses a live /connect session's SSH master when talking to
that host, and never overwrites an existing remote config (it prints the snippet
to merge instead). Driver-side only - it adds no model tool.
"""
import os
import shlex
import socket
import subprocess
from pathlib import Path


def _lan_ips():
    """This box's LAN IPv4 addresses, primary first. Best-effort, never raises."""
    ips = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("192.0.2.1", 1))      # sends nothing; just learns the route
        ips.append(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        out = subprocess.run(["ip", "-o", "-4", "addr", "show", "scope", "global"],
                             capture_output=True, text=True, timeout=3).stdout
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 4:
                ip = parts[3].split("/")[0]
                if ip not in ips:
                    ips.append(ip)
    except (OSError, ValueError):
        pass
    return ips


def _multiselect(title, rows, prompt=input):
    """Line-based multi-select (no termios, works over any terminal). `rows` is
    [(key, label)]. Returns the chosen-key set, or None if cancelled."""
    chosen = set()
    while True:
        print(title)
        for i, (key, label) in enumerate(rows, 1):
            print(f"  {i}) [{'x' if key in chosen else ' '}] {label}")
        print("  toggle by number (space/comma separated), Enter to confirm, q to cancel")
        try:
            sel = prompt("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if sel.lower() == "q":
            return None
        if not sel:
            return chosen
        for tok in sel.replace(",", " ").split():
            if tok.isdigit() and 1 <= int(tok) <= len(rows):
                k = rows[int(tok) - 1][0]
                chosen.discard(k) if k in chosen else chosen.add(k)


def _build_config(model, addrs, lean_tools):
    """Render a starter remote config.toml. `addrs` is [(name, url)] (name may
    equal url for an unnamed one); `lean-tools` is a list of enabled lean-tool names.
    Top-level keys come before any [table] (TOML requires it)."""
    named = [(n, u) for n, u in addrs if n != u]
    lines = []
    if len(addrs) == 1:
        lines.append(f'host = "{addrs[0][0]}"')
    elif addrs:
        lines.append("hosts = [" + ", ".join(f'"{n}"' for n, _ in addrs) + "]")
    lines.append(f'model = "{model}"')
    if lean_tools:
        lines.append("lean_tools_enabled = [" + ", ".join(f'"{p}"' for p in lean_tools) + "]")
    if named:
        lines.append("")
        lines.append("[machines]")
        lines += [f'"{n}" = "{u}"' for n, u in named]
    return "\n".join(lines) + "\n"


def setup(lc, cfg):
    register_command = lc["register_command"]

    def _provision(agent, cfg, arg):
        ask, dim, green = lc["_ask"], lc["dim"], lc["green"]
        yellow, red = lc["yellow"], lc["red"]

        # 1. target host: arg > connected host > pick a machine > prompt
        host = arg.strip() or (agent.remote.host if agent.remote else "")
        if not host and cfg.machines:
            rows = [(u, f"{n}  {dim(u)}") for n, u in cfg.machines.items()]
            print("which box?")
            for i, (_, label) in enumerate(rows, 1):
                print(f"  {i}) {label}")
            try:
                sel = input("number, or type a user@host: ").strip()
            except (EOFError, KeyboardInterrupt):
                print(); return
            host = rows[int(sel) - 1][0] if (sel.isdigit() and 1 <= int(sel) <= len(rows)) else sel
        if not host:
            try:
                host = input("target host (user@host): ").strip()
            except (EOFError, KeyboardInterrupt):
                print(); return
        if not host:
            print(dim("no host given; cancelled.")); return

        # Open ONE ssh master for the whole wizard so password-auth prompts once;
        # reuse a live /connect master to this host if we already have one.
        own_master = False
        if agent.remote and agent.remote.host == host:
            ctl = agent.remote.ctl
        else:
            ctl = lc["_ctl_path"](host)
            lc["_clear_master"](host, ctl)
            if subprocess.run(lc["_ssh_master_argv"](host, ctl)).returncode != 0:
                print(red(f"ssh to {host} failed (connectivity / auth)")); return
            own_master = True

        def rsh(cmd):                       # all calls reuse the master (no re-auth)
            return lc["_ssh_run_argv"](host, ctl, cmd)

        try:
            script = Path(lc["__file__"]).resolve()
            install_sh = script.parent / "install.sh"

            # 2. install lean_coder (lean: --no-ollama/--no-expose, so no Ollama
            # install or interactive network-bind + sudo). Hand install.sh our
            # master via LEANCODER_SSH_CONTROL so it reuses the same connection -
            # the whole provision is then a single password prompt.
            if ask(f"install lean_coder on {host}?"):
                if not install_sh.is_file():
                    print(yellow(f"install.sh not found next to {script.name} - reinstall "
                                 "lean_coder to enable remote install; skipping install step."))
                else:
                    env = dict(os.environ, LEANCODER_SSH_CONTROL=ctl)
                    rc = subprocess.run(["bash", str(install_sh), "--remote", host,
                                         "-y", "--no-ollama", "--no-expose"], env=env).returncode
                    if rc != 0:
                        print(red(f"install failed (exit {rc}); aborting.")); return
                    print(green(f"installed lean_coder on {host}"))

            # 3. pick local lean-tools to copy over
            mgr = lc["LeanToolManager"](lc["_lean_tools_dir"](cfg))
            sent = []
            names = [n for n in mgr.names() if n != "provision"]   # don't ship ourselves
            if names:
                sel = _multiselect("lean-tools to send:", [(n, n) for n in names])
                if sel is None:
                    print(dim("cancelled.")); return
                for name in sorted(sel):
                    p = mgr.get(name)["path"]
                    cmd = ("mkdir -p ~/.config/leancoder/lean-tools && cat > "
                           "~/.config/leancoder/lean-tools/" + shlex.quote(Path(p).name))
                    with open(p, "rb") as fh:
                        if subprocess.run(rsh(cmd), stdin=fh).returncode == 0:
                            sent.append(name)
                print(green(f"copied {len(sent)} lean-tool(s): {', '.join(sent) or 'none'}"))

            # 4. pick addresses to seed (your [machines] + this box's LAN IPs)
            rows = [(u, f"{n}  {dim(u)}") for n, u in cfg.machines.items()]
            url_name = {u: n for n, u in cfg.machines.items()}
            for ip in _lan_ips():
                u = f"http://{ip}:11434"
                if u not in url_name:
                    rows.append((u, f"this box  {dim(u)}"))
            addrs = []
            if rows:
                sel = _multiselect("addresses to seed on the remote:", rows)
                if sel is None:
                    print(dim("cancelled.")); return
                addrs = [(url_name.get(u, u), u) for u in sel]

            # 5. seed config - write only if the box has none (never clobber)
            if addrs or sent:
                cfg_text = _build_config(cfg.model, addrs, sent)
                chk = subprocess.run(rsh("cat ~/.config/leancoder/config.toml 2>/dev/null || true"),
                                     capture_output=True, text=True)
                if chk.stdout.strip():
                    print(yellow(f"{host} already has a config - not overwriting. Merge these if you want:"))
                    print(dim(cfg_text))
                else:
                    cmd = "mkdir -p ~/.config/leancoder && cat > ~/.config/leancoder/config.toml"
                    if subprocess.run(rsh(cmd), input=cfg_text, text=True).returncode == 0:
                        print(green(f"seeded {host}'s config"))

            print(green(f"provisioned {host}."))
            # a provisioned box is exactly what /connect's menu is for - offer to
            # remember it so `/connect` lists it (no-op if already saved).
            if (host not in cfg.connect_hosts
                    and host not in cfg.connect_hosts.values()
                    and ask(f"add {host} to your /connect list?")):
                cfg.connect_hosts[host] = host
                lc["save_config"](cfg)
                print(green(f"added {host} to /connect"))
        finally:
            if own_master:                  # close the master we opened
                subprocess.run(["ssh", "-o", f"ControlPath={ctl}", "-O", "exit", host],
                               capture_output=True)

    register_command("/provision", _provision,
                     "set up lean_coder on another box (install + lean-tools + addresses)")

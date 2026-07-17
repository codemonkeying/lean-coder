# lean-coder demo tape (vhs). Render: vhs demos/demo.tape
# Drives the REAL agent on a throwaway sandbox (/tmp/demo-proj) with local ollama
# (devstral:24b, thinking off) via a localhost forward so no LAN IP leaks into the GIF.
# Setup (cd + launch + a warm-up task) runs UNDER Hide so the GIF opens already in a
# live session with prior activity onscreen, not on an empty shell.
Output demos/demo.gif

Set Shell bash
Set FontSize 15
Set Width 1200
Set Height 720
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 60ms

Env OLLAMA_HOST "http://localhost:11434"

# ---- off-camera setup: enter sandbox, launch agent, do a warm-up edit ----
Hide
Type "cd /tmp/demo-proj"
Enter
Sleep 500ms
Type "lean_coder --provider ollama --model qwen2.5-coder:14b --host http://localhost:11434 --no-think --cwd . --auto --no-autosave"
Enter
Sleep 6s
Type "Read cli.py and tell me in one line what it does."
Enter
Sleep 35s
Show

# ---- on-camera: the real task, typed in human bursts ----
Sleep 1s
Type@58ms "Add a --version flag to cli.py "
Sleep 400ms
Type@58ms "that prints 1.0.0, "
Sleep 300ms
Type@58ms "then run it to prove it works."
Sleep 500ms
Enter
Sleep 40s

# Show the context footprint (the actual selling point).
Type@70ms "/ctx"
Sleep 400ms
Enter
Sleep 4s

Type@70ms "/quit"
Sleep 300ms
Enter
Sleep 2s

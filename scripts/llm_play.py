"""Let a local LLM (Ollama) drive a game through hullnes's session tools.

    uv run scripts/llm_play.py ROM "goal text" [--model gemma4:e4b] [--turns 40] [--boot 260]

The model gets tools to step the machine (with buttons), read RAM, look at the screen and ask
which instruction last wrote an address (`why`, one attention query into the transformer's
memory). The current savestate advances with each step; every state is kept so the model can
branch back to an earlier one. A transcript goes to tmp/llm/<timestamp>/transcript.md.
"""

import argparse
import base64
import json
import time
import urllib.request
from pathlib import Path

from hullnes.session import _run, describe, load, parse_addrs, ram_dump, save

TOOLS = [
    {"type": "function", "function": {
        "name": "step", "description": "Run the game for some frames with buttons held, from "
        "the current savestate (or one named by `from_state`). Returns registers and, for each "
        "address in `why`, which instruction last wrote it during the step.",
        "parameters": {"type": "object", "properties": {
            "frames": {"type": "integer", "description": "frames to run (1-300)"},
            "buttons": {"type": "string", "description": "'right' holds right for the whole step; "
                        "'right+b' runs; 'right@0-60,a@10-30' uses frame ranges counted from "
                        "the step start. Buttons: a b select start up down left right"},
            "why": {"type": "string", "description": "hex RAM addresses or ranges, e.g. "
                    "'0086,0090-0093' (work RAM is 0000-07FF; at most 32)"},
            "from_state": {"type": "string", "description": "optional earlier state name"}},
            "required": ["frames"]}}},
    {"type": "function", "function": {
        "name": "ram", "description": "Read RAM of the current savestate.",
        "parameters": {"type": "object", "properties": {
            "ranges": {"type": "string", "description": "space-separated hex addresses or "
                       "ranges, e.g. '0080-009F 075A'"}}, "required": ["ranges"]}}},
    {"type": "function", "function": {
        "name": "look", "description": "See the current screen (an image).",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "finish", "description": "End the session with your final answer.",
        "parameters": {"type": "object", "properties": {
            "answer": {"type": "string"}}, "required": ["answer"]}}},
]


def chat(model, messages, host):
    req = urllib.request.Request(f"{host}/api/chat", data=json.dumps(
        {"model": model, "messages": messages, "tools": TOOLS, "stream": False}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["message"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("rom")
    p.add_argument("goal")
    p.add_argument("--model", default="gemma4:e4b")
    p.add_argument("--turns", type=int, default=40)
    p.add_argument("--boot", type=int, default=260, help="frames to run after Start before the "
                   "model takes over")
    p.add_argument("--boot-buttons", default="start@40-45", help="inputs during --boot")
    p.add_argument("--auto-look", action="store_true", help="attach the screen to every step")
    p.add_argument("--host", default="http://localhost:11434")
    a = p.parse_args()
    out = Path("tmp/llm") / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "transcript.md").open("w")

    def note(s):
        print(s, flush=True)
        log.write(s + "\n\n")
        log.flush()

    sv, _ = _run(a.rom, None, None, a.boot, a.boot_buttons)
    states = {"s0": out / "s0.pkl"}
    save(sv, states["s0"], out / "s0.png")
    cur = "s0"
    sys_prompt = ("You are exploring an NES game that runs inside a transformer. Use the tools; "
                  "keep each step short (30-120 frames) and check results. The game state "
                  f"after boot is '{cur}': {describe(sv)}. Goal: {a.goal}")
    messages = [{"role": "system", "content": sys_prompt},
                {"role": "user", "content": "Begin."}]
    note(f"# {a.model}: {a.goal}\n\nstart: {describe(sv)}")
    for turn in range(a.turns):
        msg = chat(a.model, messages, a.host)
        messages.append({k: msg[k] for k in ("role", "content", "tool_calls") if k in msg})
        if msg.get("content"):
            note(f"**model:** {msg['content']}")
        calls = msg.get("tool_calls") or []
        if not calls:
            messages.append({"role": "user", "content": "Use a tool, or call finish."})
            continue
        for call in calls:
            fn, args = call["function"]["name"], call["function"].get("arguments") or {}
            note(f"**call** `{fn}({json.dumps(args)})`")
            if fn == "finish":
                note(f"## answer\n\n{args.get('answer', '')}")
                return
            images = []
            try:
                if fn == "step":
                    base = args.get("from_state") or cur
                    old = load(states[base])
                    why = parse_addrs(args.get("why", ""))
                    frames = max(1, min(300, int(args.get("frames", 30))))
                    sv, writers = _run(old["rom"], old["state"], old["host"], frames,
                                       args.get("buttons", ""), why)
                    cur = f"s{len(states)}"
                    states[cur] = out / f"{cur}.pkl"
                    save(sv, states[cur], out / f"{cur}.png")
                    res = f"state {cur} (from {base}): {describe(sv)}\n" + "\n".join(
                        f"${k:04X} = {v}" for k, v in writers.items())
                    if a.auto_look:
                        images = [base64.b64encode((out / f"{cur}.png").read_bytes()).decode()]
                        res += "\n(screen attached)"
                elif fn == "ram":
                    res = ram_dump(load(states[cur]), str(args.get("ranges", "")).split())
                elif fn == "look":
                    png = out / f"{cur}.png"
                    images = [base64.b64encode(png.read_bytes()).decode()]
                    res = f"screen of {cur} attached"
                else:
                    res = f"unknown tool {fn}"
            except Exception as e:  # report errors back to the model
                res = f"error: {e}"
            note(f"**result:**\n```\n{res}\n```")
            m = {"role": "tool", "content": res}
            if images:
                m["images"] = images
            messages.append(m)
    note("(turn limit reached)")


if __name__ == "__main__":
    main()

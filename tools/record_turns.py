#!/usr/bin/env python3
"""Diagnostic capture: microphone -> Universal-3.5 Pro Realtime -> raw Turn JSON.

Writes the server's message VERBATIM. No normalisation, no field filtering.
Whatever AssemblyAI sends is what lands on disk, so the schema can be
inspected rather than assumed.

    pip install websocket-client sounddevice numpy
    export ASSEMBLYAI_API_KEY=...

    # one arm, reading the script list, prompting you per utterance
    python3 tools/record_turns.py --arm A1 --script tools/diagnostic_20.txt

    # with contextual prompt / keyterms / agent context
    python3 tools/record_turns.py --arm A2 --script tools/diagnostic_20.txt
    python3 tools/record_turns.py --arm A3 --script tools/diagnostic_20.txt
    python3 tools/record_turns.py --arm A4 --script tools/diagnostic_20.txt

Output: eval/fixtures/<arm>/<NNN>.json  — one file per utterance, containing
        {"meta": {...}, "messages": [<raw server messages, in order>]}
"""
from __future__ import annotations

import argparse, json, os, queue, sys, threading, time
from pathlib import Path
from urllib.parse import urlencode

try:
    import websocket
except ImportError:
    sys.exit("pip install websocket-client")

ENDPOINT = "wss://streaming.assemblyai.com/v3/ws"
SR, CHUNK = 16000, 1600          # 16 kHz, 100 ms frames

# ---------------------------------------------------------------------------
# The four AssemblyAI arms. Kept strictly separate — prompt, keyterms and
# agent_context are different features and are measured independently.
# Limits per the current API reference: prompt 1750 chars, keyterms 100 terms,
# agent_context 1750 chars. All three are universal-3-5-pro only.
# ---------------------------------------------------------------------------
DOMAIN_PROMPT = (
    "This is an industrial maintenance inspection. The technician reports "
    "torque measurements in newton metres, pressure measurements in PSI, "
    "fluid levels as percentages, equipment serial numbers and unit "
    "identifiers such as A-104 or P-207."
)
KEYTERMS = [
    "torque", "newton metres", "Nm", "pressure", "PSI", "level", "percent",
    "serial number", "A-104", "A-140", "B-104", "P-207", "P-217", "P-208",
]
AGENT_CONTEXT = "Please confirm the serial number. Was it A-104 or A-140?"

ARMS = {
    "A1": {},                                            # base
    "A2": {"prompt": DOMAIN_PROMPT},                     # + contextual prompt
    "A3": {"prompt": DOMAIN_PROMPT,
           "keyterms_prompt": KEYTERMS},                 # + keyterms
    "A4": {"prompt": DOMAIN_PROMPT,
           "keyterms_prompt": KEYTERMS,
           "agent_context": AGENT_CONTEXT},              # + agent context
}


def build_url(arm: str, extra: dict | None = None) -> str:
    cfg = dict(ARMS[arm])
    cfg.update(extra or {})
    keyterms = cfg.pop("keyterms_prompt", None)
    params = {
        "speech_model": "universal-3-5-pro",
        "encoding": "pcm_s16le",
        "sample_rate": SR,
        **{k: v for k, v in cfg.items()},
    }
    # NOTE: format_turns is Universal-only. On Pro, turn_is_formatted tracks
    # end_of_turn, so it is deliberately not sent here.
    if keyterms:
        if len(keyterms) > 100:
            raise SystemExit("keyterms_prompt accepts at most 100 terms")
        too_long = [t for t in keyterms if len(t) > 50]
        if too_long:
            raise SystemExit(f"keyterms_prompt terms must be <=50 chars: {too_long}")
        # Per AssemblyAI docs: sent as ONE query param holding a JSON array,
        # not repeated params — repeated raw params silently drop/garble
        # multi-word terms (e.g. "newton metres") in the query string.
        params["keyterms_prompt"] = json.dumps(keyterms)
    qs = urlencode(params)
    return f"{ENDPOINT}?{qs}"


def record_one(arm: str, key: str, label: str, seconds: float,
               out: Path, extra: dict | None = None) -> dict:
    import sounddevice as sd

    messages: list[dict] = []
    audio_q: "queue.Queue[bytes|None]" = queue.Queue()
    done = threading.Event()
    t0 = time.time()

    def on_open(ws):
        def pump():
            while not done.is_set():
                c = audio_q.get()
                if c is None:
                    break
                try:
                    ws.send(c, websocket.ABNF.OPCODE_BINARY)
                except Exception:
                    break
        threading.Thread(target=pump, daemon=True).start()

    def on_message(ws, raw):
        msg = json.loads(raw)
        msg["_recv_ms"] = round((time.time() - t0) * 1000, 1)   # local timing only
        messages.append(msg)

    ws = websocket.WebSocketApp(
        build_url(arm, extra), header={"Authorization": key},
        on_open=on_open, on_message=on_message,
        on_error=lambda w, e: print(f"  ! websocket error: {e}", file=sys.stderr),
    )
    threading.Thread(target=ws.run_forever, daemon=True).start()
    time.sleep(1.2)                                   # let the socket open

    def cb(indata, frames, tinfo, status):
        if not done.is_set():
            audio_q.put(bytes(indata))

    with sd.RawInputStream(samplerate=SR, blocksize=CHUNK, channels=1,
                           dtype="int16", callback=cb):
        time.sleep(seconds)

    done.set(); audio_q.put(None)
    try:
        ws.send(json.dumps({"type": "Terminate"}))
        time.sleep(1.2)
    except Exception:
        pass
    ws.close()

    payload = {
        "meta": {
            "arm": arm, "label": label, "sample_rate": SR,
            "speech_model": "universal-3-5-pro",
            "config": {k: v for k, v in ARMS[arm].items()},
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "messages": messages,          # VERBATIM, in arrival order
    }
    out.write_text(json.dumps(payload, indent=2))
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--script", required=True, help="one utterance per line")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--speaker", default="s1")
    ap.add_argument("--start", type=int, default=1)
    args = ap.parse_args()

    key = os.environ.get("ASSEMBLYAI_API_KEY")
    if not key:
        sys.exit("ASSEMBLYAI_API_KEY is not set — refusing to run.")

    lines = [l.strip() for l in Path(args.script).read_text().splitlines()
             if l.strip() and not l.startswith("#")]
    outdir = Path("eval/fixtures") / args.arm
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\narm {args.arm} · {len(lines)} utterances · speaker {args.speaker}")
    print(f"config: {json.dumps(ARMS[args.arm]) if ARMS[args.arm] else '(base, no context)'}\n")

    for i, line in enumerate(lines[args.start - 1:], start=args.start):
        print(f"[{i:03d}/{len(lines)}] SAY:  {line}")
        input("        press Enter, then speak…")
        path = outdir / f"{args.speaker}_{i:03d}.json"
        p = record_one(args.arm, key, line, args.seconds, path)
        finals = [m for m in p["messages"]
                  if m.get("type") == "Turn" and m.get("end_of_turn")]
        got = finals[-1].get("transcript") if finals else "(no final turn)"
        print(f"        HEARD: {got}")
        print(f"        -> {path}\n")

    print(f"done. {args.arm} written to {outdir}")


if __name__ == "__main__":
    main()

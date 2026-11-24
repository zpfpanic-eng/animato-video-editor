#!/usr/bin/env python3
"""
Narration for the Animato cloud renderer and the AI Video Editor: Kyutai Pocket TTS, on this runner's CPU.

Pocket TTS is the same engine as the "Fast" voices of the Text to speech studio. It runs here, in the render
itself, so no second runner and no outside speech service is needed.

Pocket TTS does not report where each word is spoken, so the word timings are worked out per sentence from the
real length of each sentence's audio (words share the sentence's time by their length). That is close enough to
drive the karaoke captions and scene timing.

usage: tts.py --text-file script.txt --voice alba \
              --rate +0% --out-audio narration.mp3 --out-words words.json

--voice  a Pocket voice name (alba, jane, vera, marius, michael, george, ...). Old voice names such as
         en-US-AvaMultilingualNeural are mapped to a Pocket voice. Languages Pocket TTS cannot speak stop with exit code 6.
--rate   speaking speed as a percentage, "+5%" = 5% faster, "-3%" = 3% slower (applied with ffmpeg).
--pitch  accepted and ignored (kept so older callers do not break).
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import wave

FEMALE = ["alba", "jane", "eve", "vera", "mary", "anna", "caro_davy", "cosette", "eponine", "fantine", "azelma"]
MALE = ["marius", "michael", "george", "charles", "paul", "jean", "javert", "bill_boerst", "peter_yearsley", "stuart_bell"]
KNOWN = set(FEMALE + MALE)
FEMALE_NAMES = ("ava", "emma", "aria", "jenny", "jane", "sara", "libby", "sonia", "michelle", "ana", "clara", "natasha", "neerja", "ezinne", "female")
# The old English voice names keep their own Pocket voice, so a scene with several characters still sounds like several people.
EN_MAP = {"ava": "alba", "emma": "jane", "aria": "vera", "jenny": "eve", "sonia": "mary", "michelle": "anna", "ezinne": "cosette",
          "andrew": "marius", "brian": "michael", "guy": "george", "christopher": "charles", "ryan": "paul", "eric": "jean", "abeo": "javert"}
# Other languages Pocket TTS has a ready voice for. Every other language is refused (it would read the text in an English voice).
LANG_VOICES = {"fr": "estelle", "es": "lola", "it": "giovanni", "de": "juergen", "pt": "rafael"}
GAP_SECONDS = 0.18          # short pause after each sentence
CHUNK_CHARS = 320           # longest piece sent to the engine at once


def pick_voice(voice: str):
    """A Pocket voice name for whatever voice the caller asked for; None when Pocket TTS cannot speak that language."""
    v = (voice or "").strip().lower()
    if not v:
        return "alba"
    if v in KNOWN:
        return v
    m = re.fullmatch(r"([a-z]{2,3})-([a-z]{2,4})-([a-z]+?)(?:multilingual)?neural", v)     # e.g. en-us-avamultilingualneural
    if m:
        lang, name = m.group(1), m.group(3)
        if lang == "en":
            if name in EN_MAP:
                return EN_MAP[name]
            pool = FEMALE if any(n in name for n in FEMALE_NAMES) else MALE
            return pool[sum(map(ord, name)) % len(pool)]
        return LANG_VOICES.get(lang)
    return "alba" if any(n in v for n in FEMALE_NAMES) else "marius"


def split_pieces(text: str):
    """Sentences, kept short enough for the engine; every piece keeps its own words."""
    sentences = re.split(r"(?<=[.!?…])\s+|\n+", text.strip())
    pieces = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        while len(s) > CHUNK_CHARS:
            cut = max(s.rfind(",", 0, CHUNK_CHARS), s.rfind(";", 0, CHUNK_CHARS), s.rfind(" ", 0, CHUNK_CHARS))
            cut = cut if cut > 40 else CHUNK_CHARS
            pieces.append(s[:cut + 1].strip())
            s = s[cut + 1:].strip()
        if s:
            pieces.append(s)
    return pieces


def parse_rate(value: str) -> float:
    """"+5%" -> 1.05, "-3%" -> 0.97. Anything unreadable is normal speed. ffmpeg's atempo allows 0.5 to 2."""
    m = re.fullmatch(r"\s*([+-]?\d{1,3})\s*%?\s*", str(value or ""))
    if not m:
        return 1.0
    return max(0.5, min(2.0, 1.0 + int(m.group(1)) / 100.0))


def write_wav(path: str, samples, rate: int):
    import numpy as np
    a = np.clip(np.asarray(samples, dtype="float32").reshape(-1), -1.0, 1.0)
    pcm = (a * 32767.0).astype("<i2").tobytes()
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(rate))
        w.writeframes(pcm)


def main():
    # Accept "--rate -3%" as well as "--rate=-3%" (argparse would read "-3%" as an option).
    argv, i, fixed = sys.argv[1:], 0, []
    while i < len(argv):
        if argv[i] in ("--rate", "--pitch") and i + 1 < len(argv):
            fixed.append(f"{argv[i]}={argv[i + 1]}")
            i += 2
            continue
        fixed.append(argv[i])
        i += 1
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-file", required=True)
    ap.add_argument("--voice", default="alba")
    ap.add_argument("--rate", default="+0%")
    ap.add_argument("--pitch", default="+0Hz")
    ap.add_argument("--out-audio", required=True)
    ap.add_argument("--out-words", required=True)
    args = ap.parse_args(fixed)

    with open(args.text_file, encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        print("empty text", file=sys.stderr)
        sys.exit(2)

    voice = pick_voice(args.voice)
    if voice is None:
        print(f"Pocket TTS has no voice for {args.voice} (languages: English, French, Spanish, Italian, German, Portuguese)", file=sys.stderr)
        sys.exit(6)
    tempo = parse_rate(args.rate)
    pieces = split_pieces(text)

    try:
        import numpy as np
        from pocket_tts import TTSModel        # imported here so --help works without the package
    except ImportError as err:
        print(f"{type(err).__name__}: {err}", file=sys.stderr)   # contains "No module named", which the callers recognise
        sys.exit(4)

    try:
        model = TTSModel.load_model()
        rate = int(model.sample_rate)
        state = model.get_state_for_audio_prompt(voice)
        chunks, spans = [], []            # audio of each piece, and (words, seconds) of each piece
        gap = np.zeros(int(rate * GAP_SECONDS), dtype="float32")
        for piece in pieces:
            audio = model.generate_audio(state, piece)
            a = np.asarray(audio.numpy() if hasattr(audio, "numpy") else audio, dtype="float32").reshape(-1)
            if a.size == 0:
                continue
            chunks.append(a)
            chunks.append(gap)
            spans.append((piece.split(), (a.size + gap.size) / rate))
        if not chunks:
            raise RuntimeError("the voice produced no sound")
        samples = np.concatenate(chunks)
    except Exception as err:  # noqa: BLE001 - report any engine error clearly
        print(f"pocket-tts failed: {type(err).__name__}: {err}", file=sys.stderr)
        sys.exit(3)

    # Word timings: each sentence's real time shared by its words, by length. Spoken faster, every time shrinks.
    words, t = [], 0.0
    for toks, secs in spans:
        weights = [len(w) + 1 for w in toks]
        total = float(sum(weights)) or 1.0
        speech = max(0.05, secs - GAP_SECONDS)
        at = t
        for w, wt in zip(toks, weights):
            d = speech * wt / total
            words.append({"text": w, "start": at / tempo, "end": (at + d) / tempo})
            at += d
        t += secs

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = os.path.join(tmp, "voice.wav")
        write_wav(wav_path, samples, rate)
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", wav_path]
        if abs(tempo - 1.0) > 0.005:
            cmd += ["-filter:a", f"atempo={tempo:.4f}"]
        cmd += ["-ar", "44100", "-ac", "1", "-codec:a", "libmp3lame", "-b:a", "128k", args.out_audio]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(args.out_audio):
            print(f"ffmpeg failed: {r.stderr.strip()[-300:]}", file=sys.stderr)
            sys.exit(5)

    with open(args.out_words, "w", encoding="utf-8") as f:
        json.dump({"voice": voice, "engine": "pocket-tts", "words": words, "sentences": []}, f)
    print(f"{voice}: {len(words)} word timings (from sentence lengths), {len(pieces)} sentence(s)")


if __name__ == "__main__":
    main()

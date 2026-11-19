"""Prepare continuous 40 s mono excerpts and record provenance and statistics."""
import argparse
import csv
import io
from math import gcd
from pathlib import Path
import zipfile

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from fetch import NOISEX, RAW, SONYC, SONYC_NAMES, digest

FS = 16000
SECONDS = 40
DATA = Path(__file__).resolve().parent
NOISEX_CLASSES = ("machinegun", "factory1", "destroyerengine", "volvo")


def mono(wav):
    if wav.dtype.kind in "iu":
        bits = wav.dtype.itemsize * 8
        wav = wav.astype(np.float64) / (2 ** (bits - 1)) - (1 if wav.dtype.kind == "u" else 0)
    else:
        wav = wav.astype(np.float64)
    return wav if wav.ndim == 1 else wav.mean(axis=1)


def metrics(x):
    centered = x - np.mean(x)
    variance = np.mean(centered ** 2)
    rms = np.sqrt(np.mean(x ** 2))
    power = np.mean(x[:len(x) // 128 * 128].reshape(-1, 128) ** 2, axis=1)
    return dict(kurtosis=float(np.mean(centered ** 4) / variance ** 2) if variance > 0 else None,
                crest_factor=float(np.max(np.abs(x)) / rms) if rms > 0 else None,
                power_cv=float(np.std(power) / np.mean(power)) if np.any(power) else None,
                zero_power_blocks=int(np.count_nonzero(power == 0)),
                blocks=len(power))


def process(name, provenance, source, dest):
    row = dict(name=name, source=provenance)
    target = dest / f"{name}.wav"
    if source is None:
        return dict(row, status="missing")
    fs, samples = wavfile.read(source)
    end = min(len(samples), SECONDS * fs)
    x = mono(samples[:end])
    divisor = gcd(fs, FS)
    if fs != FS:
        x = resample_poly(x, FS // divisor, fs // divisor)
    x = x.astype("<f4")
    eligible = len(x) >= SECONDS * FS and np.any(x)
    if eligible:
        wavfile.write(target, FS, x)
    else:
        target.unlink(missing_ok=True)
    return dict(row, status="prepared" if eligible else ("short" if len(x) < SECONDS * FS else "silent"),
                source_fs=fs, source_channels=1 if samples.ndim == 1 else samples.shape[1],
                recording_s=len(samples) / fs, start_sample=0, end_sample_exclusive=end,
                seconds=len(x) / FS, output=target.name if eligible else "", **metrics(x))


VALVE_NAMES = tuple(f"{i:08d}.wav" for i in range(4))


def join_clips(name, clips, dest):
    """Crossfade four ten-second clips and restore the 30 ms overlap loss."""
    if len(clips) != 4:
        raise ValueError("Expected four ten-second clips")
    fade = FS // 100
    weight = np.linspace(0, 1, fade)
    sources = " | ".join(label for _, label in clips)
    if not all(path.exists() for path, _ in clips):
        return dict(name=name, source=sources, status="missing")
    parts, seams = [], []
    for path, _ in clips:
        rate, wav = wavfile.read(path)
        if len(wav) < 10 * rate:
            raise ValueError(f"Clip shorter than 10 s: {path}")
        x = mono(wav[:10 * rate])
        divisor = gcd(rate, FS)
        x = resample_poly(x, FS // divisor, rate // divisor) if rate != FS else x
        parts.append(x)
    joined = parts[0]
    for x in parts[1:]:
        start = len(joined) - fade
        seams.append((start, start + fade))
        blend = joined[-fade:] * (1 - weight) + x[:fade] * weight
        joined = np.concatenate((joined[:-fade], blend, x[fade:]))
    target_length = 40 * FS
    divisor = gcd(len(joined), target_length)
    output = resample_poly(joined, target_length // divisor, len(joined) // divisor).astype("<f4")
    if len(output) != target_length:
        raise ValueError("Unexpected joined length")
    target = dest / f"{name}.wav"
    wavfile.write(target, FS, output)
    final_seams = " | ".join(f"{round(lo * target_length / len(joined))}:{round(hi * target_length / len(joined))}"
                             for lo, hi in seams)
    return dict(name=name, source=sources, status="prepared_joined", recording_s=40,
                seconds=len(output) / FS, output=target.name, source_intervals_s="[0,10) each",
                crossfade_samples=fade, seams_16k=" | ".join(f"{lo}:{hi}" for lo, hi in seams),
                output_seams_16k=final_seams, **metrics(output))


def generate(dest):
    dest.mkdir(parents=True, exist_ok=True)
    rows = []
    archive = RAW / NOISEX[0]
    names = {name: f"NOISEX92/{name}/signal.wav" for name in NOISEX_CLASSES}
    if archive.exists():
        if digest(archive, "md5") != NOISEX[2]:
            raise ValueError("NOISEX-92 archive MD5 mismatch")
        with zipfile.ZipFile(archive) as z:
            for name, member in names.items():
                rows.append(process(name, member, io.BytesIO(z.read(member)), dest))
    else:
        rows.extend(process(name, member, None, dest) for name, member in names.items())
    valve = [(RAW / "mimii" / name, f"6_dB_valve.zip:valve/id_00/normal/{name}") for name in VALVE_NAMES]
    rows.append(join_clips("valve", valve, dest))
    hammer = [(RAW / "sonyc" / name, f"{SONYC[0]}:audio-0/{name}") for name in SONYC_NAMES]
    rows.append(join_clips("jackhammer", hammer, dest))
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with (dest / "noise.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"{sum(r['status'].startswith('prepared') for r in rows)} prepared excerpts; {dest / 'noise.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DATA / "prepared")
    generate(parser.parse_args().output)

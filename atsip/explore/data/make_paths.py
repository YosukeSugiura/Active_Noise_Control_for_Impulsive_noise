"""Generate deterministic 16 kHz MeshRIR path pairs and a JSON ledger."""
import argparse
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from scipy.signal import resample_poly

from fetch import MESH, RAW, digest

FS = 16000
DISTANCES_CM = (5, 10, 20, 40)
PREFIX = "S32-M441_npy/"
DATA = Path(__file__).resolve().parent


def load(archive, stem):
    return np.load(io.BytesIO(archive.read(f"{PREFIX}{stem}.npy")), allow_pickle=False)


def propagation(mic, sources):
    return np.linalg.norm(sources - mic, axis=1) * FS / 343.0


def peaks(irs, centers):
    result = []
    for row, center in zip(irs, centers):
        sample = round(center)
        lo, hi = sample - 12, sample + 13
        if lo < 0 or hi > len(row):
            raise ValueError("Direct sound window outside the response")
        result.append(lo + int(np.argmax(np.abs(row[lo:hi]))))
    return np.asarray(result)


def select(delays, mics, count):
    candidates = []
    for e, row in enumerate(delays):
        for a, p in enumerate(row):
            for b in np.flatnonzero(row < p):
                candidates.append((int(row[b] - p), e, a, int(b)))
    used_sources, used_mics = set(), set()
    selected = []
    for _, e, a, b in sorted(candidates):
        if e in used_mics or a in used_sources or b in used_sources:
            continue
        moved = {}
        distance = np.linalg.norm(mics - mics[e], axis=1)
        for cm in DISTANCES_CM:
            neighbors = np.flatnonzero(np.isclose(distance, cm / 100, atol=0.0005, rtol=0))
            available = [int(i) for i in neighbors if i not in used_mics]
            if not available:
                break
            moved[cm] = available[0]
        if len(moved) != len(DISTANCES_CM):
            continue
        selected.append((e, a, b, moved))
        used_sources.update((a, b))
        used_mics.update((e, *moved.values()))
        if len(selected) == count:
            break
    return selected, len(candidates)


def decay_length(row, direct):
    tail = np.cumsum(np.square(row, dtype=np.float64)[::-1])[::-1]
    below = np.flatnonzero(tail[direct:] <= tail[direct] * 1e-3)
    return int(direct) + int(below[0]) if below.size else len(row)


def generate(source, dest):
    if digest(source, "md5") != MESH[2]:
        raise ValueError("MeshRIR archive MD5 mismatch")
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source) as archive:
        mics, sources = (load(archive, key) for key in ("pos_mic", "pos_src"))
        if mics.shape != (441, 3) or sources.shape != (32, 3):
            raise ValueError("Expected 441 microphones and 32 sources")
        # Median system delay estimated at grid point 0 from the 32 peak arrivals.
        first = resample_poly(load(archive, "ir_0")[:, :8192], 1, 3, axis=-1)
        offset = float(np.median(np.argmax(np.abs(first), axis=1) - propagation(mics[0], sources)))
        delays = []
        for e, mic in enumerate(mics):
            irs = resample_poly(load(archive, f"ir_{e}")[:, :8192], 1, 3, axis=-1)
            delays.append(peaks(irs, propagation(mic, sources) + offset))
        selected, candidates = select(delays, mics, 8)
        pairs = []
        for i, (e, a, b, moved) in enumerate(selected, 1):
            arrays, metadata = {}, {}
            for cm, point in [(0, e), *moved.items()]:
                irs = resample_poly(load(archive, f"ir_{point}")[[a, b]], 1, 3, axis=-1)
                direct = delays[point][[a, b]]
                if not np.array_equal(direct, peaks(irs, propagation(mics[point], sources[[a, b]]) + offset)):
                    raise ValueError("Direct arrival from the prefix and whole response disagree")
                suffix = f"_{cm}cm" if cm else ""
                arrays.update({f"P{suffix}": irs[0], f"S{suffix}": irs[1]})
                metadata[cm] = dict(grid=point, xyz=mics[point].tolist(),
                                    distance_cm=float(100 * np.linalg.norm(mics[point] - mics[e])),
                                    direct=direct.tolist(),
                                    decay=[decay_length(row, d) for row, d in zip(irs, direct)])
            length = max(max(row["decay"]) for row in metadata.values())
            arrays = {key: row[:length].astype("<f4") for key, row in arrays.items()}
            for cm in moved:
                metadata[cm]["relative_change"] = [
                    float(np.linalg.norm(arrays[f"{key}_{cm}cm"] - arrays[key]) / np.linalg.norm(arrays[key]))
                    for key in ("P", "S")]
            file = f"pair_{i:02}.npz"
            np.savez_compressed(dest / file, fs=FS, **arrays)
            pairs.append(dict(file=file, a=a, b=b, a_xyz=sources[a].tolist(), b_xyz=sources[b].tolist(),
                              length=length, measurements=metadata))
    ledger = dict(fs=FS, source_md5=MESH[2], direct_offset=offset, candidates=candidates, pairs=pairs)
    (dest / "paths.json").write_text(json.dumps(ledger, indent=2) + "\n")
    print(f"{len(pairs)}/8 pairs, {candidates} candidates; {dest}")


OFF16 = 1880
P_LENGTH = 4096
S_LENGTH = 2048


def generate16(source, dest):
    if digest(source, "md5") != MESH[2]:
        raise ValueError("MeshRIR archive MD5 mismatch")
    s0 = json.loads((DATA / "paths/paths.json").read_text())
    offset = s0["direct_offset"]
    dest.mkdir(parents=True, exist_ok=True)
    pairs, missing = [], []
    with zipfile.ZipFile(source) as archive:
        mics, sources = (load(archive, key) for key in ("pos_mic", "pos_src"))
        for pair in s0["pairs"]:
            a, b = pair["a"], pair["b"]
            original = pair["measurements"]
            arrays, measurements = {}, {}
            for cm, point in original.items():
                e = point["grid"]
                irs = resample_poly(load(archive, f"ir_{e}"), 1, 3, axis=-1)
                predicted = propagation(mics[e], sources)
                measured = peaks(irs, predicted + offset)
                if measured[[a, b]].tolist() != point["direct"]:
                    raise ValueError(f"S0 direct arrivals differ for grid {e}")

                def save_path(key, src, length):
                    arr = irs[src, OFF16:OFF16 + length].astype("<f4")
                    if len(arr) != length:
                        raise ValueError(f"Insufficient samples for {key} at grid {e}")
                    arrays[key] = arr
                    return dict(source=src, xyz=sources[src].tolist(),
                                direct=int(measured[src]) - OFF16,
                                geometry=float(predicted[src]),
                                predicted=float(predicted[src] + offset - OFF16))

                suffix = "" if cm == "0" else f"_{cm}cm"
                measurements[cm] = dict(grid=e, xyz=point["xyz"], distance_cm=point["distance_cm"],
                                        P=save_path("P" + suffix, a, P_LENGTH),
                                        S=save_path("S" + suffix, b, S_LENGTH))
                if cm != "0":
                    continue
                distances = np.linalg.norm(sources - sources[a], axis=1)
                # Rounded metric distances break numerical ties from symmetric coordinates.
                candidates = [i for i in range(len(sources)) if i not in (a, b)
                              and measured[i] >= measured[b] + 16]
                if not candidates:
                    break
                for name, sign in (("P_near", 1), ("P_far", -1)):
                    src = min(candidates, key=lambda i: (sign * round(float(distances[i]), 12), i))
                    item = save_path(name, src, P_LENGTH)
                    item.update(distance_from_a=float(distances[src]), gap=int(measured[src]-measured[b]),
                                relative_change=float(np.linalg.norm(arrays[name]-arrays["P"])
                                                      / np.linalg.norm(arrays["P"])))
                    measurements[cm][name] = item
            if "P_near" not in arrays:
                missing.append(pair["file"])
                (dest / pair["file"]).unlink(missing_ok=True)
                continue
            np.savez_compressed(dest / pair["file"], fs=FS, **arrays)
            pairs.append(dict(file=pair["file"], a=a, b=b, measurements=measurements))
    ledger = dict(fs=FS, source_md5=MESH[2], removed=OFF16, direct_offset=offset,
                  p_length=P_LENGTH, s_length=S_LENGTH, pairs=pairs, missing=missing)
    (dest / "paths16.json").write_text(json.dumps(ledger, indent=2) + "\n")
    print(f"{len(pairs)} paths16 pairs; missing: {missing}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=RAW / MESH[0])
    parser.add_argument("--paths16", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.paths16:
        generate16(args.archive, args.output or DATA / "paths16")
    else:
        generate(args.archive, args.output or DATA / "paths")

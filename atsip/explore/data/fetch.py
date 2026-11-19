"""Fetch the S0 data into raw/ and verify before publishing local files."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
from urllib.request import Request, urlopen
import zipfile

RAW = Path(__file__).resolve().parent / "raw"
ZENODO = "https://zenodo.org/records"
MESH = ("S32-M441_npy.zip", f"{ZENODO}/10852693/files/S32-M441_npy.zip?download=1",
        "9818fc66b36513590e7abd071243d8e9")
NOISEX = ("NOISEX92.zip", f"{ZENODO}/3921794/files/NOISEX92.zip?download=1",
          "b5b707b8ac7217713e3123360c21043c")
VALVE = f"{ZENODO}/3384388/files/6_dB_valve.zip?download=1"
VALVE_SHA256 = (
    "16ba2d22017d3ef66ec453300d0ee1698cd2081962bd49a274f2493d7e6f3420",
    "39bd5fcaeaabb02765307c8835518c4c22945acdef7dae33aea18e5d7a534b09",
    "71e12c854af0bdc83492e14f7b33519187f494969eeae1e37fb9311150680625",
    "42d0b6a704d77819655cf220189aae77cb0afecb604ef33a103e3b4757e51732",
)
URBAN = f"{ZENODO}/1203745/files/UrbanSound8K.tar.gz?download=1"
URBAN_INDEX = ("urbansound8k_index_1.0.json",
               f"{ZENODO}/11176928/files/urbansound8k_index_1.0.json?download=1",
               "1c4940e08c1305c49b592f3d9c103e6f")


SONYC = ("audio-0.tar.gz", f"{ZENODO}/3966543/files/audio-0.tar.gz?download=1",
         "bbb4dbae7d2e58e18d24878b9ee1eb51")
SONYC_NAMES = ("00_010346.wav", "04_002110.wav", "16_027820.wav", "37_016642.wav")


def digest(path, method):
    with path.open("rb") as source:
        return hashlib.file_digest(source, method).hexdigest()


def fetch(artifact):
    name, url, md5 = artifact
    path = RAW / name
    if path.exists():
        if digest(path, "md5") != md5:
            raise ValueError(f"MD5 mismatch: {path}")
        return path
    part = path.with_suffix(path.suffix + ".part")
    try:
        with urlopen(url, timeout=120) as source, part.open("wb") as dest:
            shutil.copyfileobj(source, dest, 1024 * 1024)
        if digest(part, "md5") != md5:
            raise ValueError(f"MD5 mismatch: {part}")
        part.replace(path)
    finally:
        part.unlink(missing_ok=True)
    return path


class RangeReader:
    """Seek in a remote ZIP without materializing the archive."""

    def __init__(self, url):
        self.url = url
        self.position = 0
        _, self.size = self.range(0, 0)

    def range(self, start, end):
        req = Request(f"{self.url}&s0_range={start}-{end}",
                      headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"})
        with urlopen(req, timeout=90) as response:
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
            if response.status != 206 or not match or tuple(map(int, match.groups()[:2])) != (start, end):
                raise OSError(f"Server did not honor bytes={start}-{end}")
            data = response.read(end - start + 1)
        if len(data) != end - start + 1:
            raise OSError("Incomplete HTTP range")
        return data, int(match[3])

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}[whence]
        self.position = base + offset
        if self.position < 0:
            raise OSError("Negative seek")
        return self.position

    def tell(self):
        return self.position

    def seekable(self):
        return True

    def read(self, n=-1):
        end = self.size if n < 0 else min(self.size, self.position + n)
        if end <= self.position:
            return b""
        data, size = self.range(self.position, end - 1)
        if size != self.size:
            raise OSError("Remote ZIP size changed")
        self.position = end
        return data


def valve():
    with zipfile.ZipFile(RangeReader(VALVE)) as archive:
        names = sorted(n for n in archive.namelist()
                       if re.search(r"(?:^|/)id_00/normal/[^/]+\.wav$", n))
        if not names:
            raise ValueError("No valve id_00/normal WAV found")
        if len(names) < 4:
            raise ValueError("Fewer than four valve recordings")
        paths = []
        for name in names[:4]:
            paths.append(extract_valve(archive, name))
        return paths


def extract_valve(archive, name):
    data = archive.read(name)  # ZipFile verifies the CRC-32 of the extracted member.
    path = RAW / "mimii" / Path(name).name
    path.parent.mkdir(exist_ok=True)
    sha256 = hashlib.sha256(data).hexdigest()
    if sha256 != VALVE_SHA256[int(path.stem)]:
        raise ValueError(f"Valve WAV SHA-256 mismatch: {name}")
    path.write_bytes(data)
    print(f"{name}: CRC-32 {archive.getinfo(name).CRC:08x}, SHA-256 {sha256}")
    return path

def urban():
    index = json.loads(fetch(URBAN_INDEX).read_text())["clips"]
    hashes = {str(PurePosixPath(*PurePosixPath(item["audio"][0]).parts[-2:])): item["audio"][1]
              for item in index.values()}
    dest = RAW / "urbansound8k"
    dest.mkdir(exist_ok=True)
    # Class IDs from the publisher's metadata. One representative per requested class.
    pending = {"3", "7"}
    found = []
    with urlopen(URBAN, timeout=120) as source, tarfile.open(fileobj=source, mode="r|gz") as archive:
        for member in archive:
            parts = PurePosixPath(member.name).parts
            fields = parts[-1].split("-")
            if not member.isfile() or not member.name.endswith(".wav") or len(fields) != 4 or fields[1] not in pending:
                continue
            relative = str(PurePosixPath(*parts[-2:]))
            data = archive.extractfile(member).read()
            if hashlib.md5(data).hexdigest() != hashes[relative]:
                raise ValueError(f"Soundata MD5 mismatch: {relative}")
            path = dest / parts[-1]
            path.write_bytes(data)
            found.append(path)
            pending.remove(fields[1])
            print(f"{member.name}: verified MD5 {hashes[relative]}", flush=True)
            if not pending:
                return found
    raise ValueError(f"UrbanSound8K class IDs missing: {pending}")


def sonyc():
    archive = fetch(SONYC)
    dest = RAW / "sonyc"
    dest.mkdir(exist_ok=True)
    pending = {f"audio-0/{name}": name for name in SONYC_NAMES}
    with tarfile.open(archive, "r|gz") as source:
        for member in source:
            if not member.isfile() or member.name not in pending:
                continue
            name = pending.pop(member.name)
            with source.extractfile(member) as audio:
                (dest / name).write_bytes(audio.read())
            if not pending:
                return dest
    raise ValueError(f"Missing SONYC-UST members: {sorted(pending)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", metavar="DATASET")
    args = parser.parse_args()
    RAW.mkdir(exist_ok=True)
    actions = {"mesh": lambda: fetch(MESH), "noisex": lambda: fetch(NOISEX),
               "mimii": valve, "urban": urban, "sonyc": sonyc}
    failed = []
    if unknown := set(args.datasets) - actions.keys():
        parser.error(f"Unknown datasets: {sorted(unknown)}")
    for name in args.datasets or ("mesh", "mimii", "noisex", "sonyc"):
        try:
            print(actions[name](), flush=True)
        except (OSError, ValueError, KeyError, zipfile.BadZipFile, tarfile.TarError) as error:
            print(f"{name}: {error}", flush=True)
            failed.append(name)
    if failed:
        raise SystemExit(f"Failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()

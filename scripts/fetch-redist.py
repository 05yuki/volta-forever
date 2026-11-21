#!/usr/bin/env python3
"""Download NVIDIA redistributable archives listed in a redistrib manifest,
verify their SHA-256 and unpack them into one merged tree.

usage: fetch-redist.py BASE_URL MANIFEST DL_DIR TREE_DIR [--skip REGEX] [--only REGEX]
                       [--platform linux-x86_64] [--variant NAME] [--runtime] [--discard]

BASE_URL   e.g. https://developer.download.nvidia.com/compute/cuda/redist
MANIFEST   a local JSON file or an https:// URL to one
DL_DIR     where the archives are kept (one file per component)
TREE_DIR   merged tree (bin/, include/, lib/, nvvm/ ...), like /usr/local/cuda
--variant  for manifests with per-CUDA variants (cuDNN: cuda12 / cuda13)
--runtime  unpack the shared libraries (lib/*.so*) and the licence alone: what a program
           needs to run, without the headers and the static libraries
--discard  remove each archive once it is unpacked; what TREE_DIR/COMPONENTS.tsv records
           as unpacked is then not downloaded again

Needs: Python 3, GNU tar. Uses aria2c when available; falls back to Python's own
urllib (slower, single-threaded). Re-runnable: verified archives are not downloaded again.
"""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _has_aria2c():
    return shutil.which("aria2c") is not None


def download(url, dest_dir, dest_name, expected_sha256, lock_fds):
    """Download url to dest_dir/dest_name, verify SHA-256. Returns the path."""
    dest = os.path.join(dest_dir, dest_name)
    if _has_aria2c():
        subprocess.run(
            ["aria2c", "-q", "-x", "8", "-s", "8", "--allow-overwrite=true",
             "--auto-file-renaming=false", "-d", dest_dir, "-o", dest_name, url],
            check=True, pass_fds=lock_fds,
        )
    else:
        # Fall back to Python's urllib: download to a temporary file, rename on success.
        import urllib.request
        tmp_fd, tmp_path = tempfile.mkstemp(dir=dest_dir, prefix=".dl-")
        try:
            with os.fdopen(tmp_fd, "wb") as tmp_f:
                with urllib.request.urlopen(url) as resp:
                    while True:
                        chunk = resp.read(1 << 22)
                        if not chunk:
                            break
                        tmp_f.write(chunk)
            os.rename(tmp_path, dest)
        except BaseException:
            os.unlink(tmp_path)
            raise
    got = sha256(dest)
    if got != expected_sha256:
        sys.exit(f"SHA-256 mismatch for {dest}: {got} != {expected_sha256}")
    return dest


def load_manifest(path_or_url):
    """Load a JSON manifest from a local file or an https:// URL."""
    if path_or_url.startswith("https://") or path_or_url.startswith("http://"):
        import urllib.request
        with urllib.request.urlopen(path_or_url) as resp:
            return json.loads(resp.read())
    return json.load(open(path_or_url))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base_url")
    ap.add_argument("manifest", help="local JSON file or https:// URL")
    ap.add_argument("dl_dir")
    ap.add_argument("tree_dir")
    ap.add_argument("--skip", default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--platform", default="linux-x86_64")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--runtime", action="store_true")
    ap.add_argument("--discard", action="store_true")
    a = ap.parse_args()

    if not _has_aria2c():
        print("note: aria2c not found; using Python's urllib (slower)", file=sys.stderr)

    manifest = load_manifest(a.manifest)
    os.makedirs(a.dl_dir, exist_ok=True)
    os.makedirs(a.tree_dir, exist_ok=True)
    rows = []
    for name, comp in sorted(manifest.items()):
        if not isinstance(comp, dict) or a.platform not in comp:
            continue
        if a.skip and re.search(a.skip, name):
            continue
        if a.only and not re.search(a.only, name):
            continue
        entry = comp[a.platform]
        if "relative_path" not in entry:  # per-variant entries
            if a.variant is None or a.variant not in entry:
                print(f"skip {name}: variants {sorted(entry)} (use --variant)", file=sys.stderr)
                continue
            entry = entry[a.variant]
        rows.append((name, comp.get("version", "?"), entry))

    # Both the merged install tree and the archive cache may be shared by fetches.
    # Lock them in a consistent order, including when two installs share just their cache.
    with contextlib.ExitStack() as locks:
        directories = sorted({os.path.realpath(a.tree_dir), os.path.realpath(a.dl_dir)})
        lock_fds = []
        for directory in directories:
            lock = locks.enter_context(open(os.path.join(directory, ".vf-fetch.lock"), "a"))
            fcntl.flock(lock, fcntl.LOCK_EX)
            lock_fds.append(lock.fileno())
        # The active child keeps the locks if the parent is killed mid-download or extraction.
        install_components(a, rows, lock_fds)
    print(f"done: {len(rows)} components -> {a.tree_dir}")


def read_receipts(path):
    receipts = {}
    try:
        with open(path) as record:
            for line in record:
                fields = line.rstrip("\n").split("\t")
                if len(fields) == 4:
                    # Older receipts guarantee the runtime subset only.
                    fields.append("runtime")
                if len(fields) == 5:
                    receipts[fields[0]] = fields
    except FileNotFoundError:
        pass
    return receipts


def write_receipts(path, receipts):
    # Writers hold the tree lock. Publish a complete snapshot so a killed fetch
    # leaves either the previous receipts or the new ones available to its retry.
    tmp = path + ".tmp"
    with open(tmp, "w") as record:
        for fields in receipts.values():
            record.write("\t".join(fields) + "\n")
        record.flush()
        os.fsync(record.fileno())
    os.replace(tmp, path)


def install_components(a, rows, lock_fds):
    record = os.path.join(a.tree_dir, "COMPONENTS.tsv")
    receipts = read_receipts(record)
    requested = "runtime" if a.runtime else "full"
    extract = ["--strip-components=1", "-C", a.tree_dir]
    if a.runtime:
        extract += ["--wildcards", "--exclude=*/lib/stubs/*", "*/lib/*.so*", "*/LICENSE"]

    # (Said before anything is downloaded: tar starts xz for these archives.)
    if any(entry["relative_path"].endswith(".xz") for _, _, entry in rows) and shutil.which("xz") is None:
        sys.exit("xz is needed to unpack NVIDIA's archives and was not found "
                 "(Debian and Ubuntu: the package xz-utils; Fedora and RHEL: xz)")

    for name, version, entry in rows:
        url = f"{a.base_url}/{entry['relative_path']}"
        dest = os.path.join(a.dl_dir, os.path.basename(entry["relative_path"]))
        expected = entry["sha256"]
        prior = receipts.get(name)
        same_archive = prior and prior[3] == expected
        if a.discard and same_archive and prior[4] in (requested, "full"):
            print(f"have {name} {version} (unpacked before)", flush=True)
            continue
        if not (os.path.exists(dest) and sha256(dest) == entry["sha256"]):
            print(f"get  {name} {version} ({int(entry['size']) / 1e6:.1f} MB)", flush=True)
            download(url, a.dl_dir, os.path.basename(dest), expected, lock_fds)
        else:
            print(f"have {name} {version}", flush=True)
        # A failed extraction may already have overwritten files of the previous
        # version. Invalidate its receipt before tar starts modifying the tree.
        if prior:
            del receipts[name]
            write_receipts(record, receipts)
        if subprocess.run(["tar", "-xf", dest] + extract, pass_fds=lock_fds).returncode != 0:
            sys.exit(f"cannot unpack {dest} into {a.tree_dir}")
        installed = "full" if same_archive and prior[4] == "full" else requested
        receipts[name] = [name, version, os.path.basename(dest), expected, installed]
        write_receipts(record, receipts)
        if a.discard:
            os.remove(dest)


if __name__ == "__main__":
    main()

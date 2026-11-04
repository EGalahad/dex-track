"""Download OakInk2 annotations and shared assets, without video."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _source import REVISION, source_root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=15, help="Number of benchmark recordings, 1..15")
    parser.add_argument("--all", action="store_true", help="Download every annotation recording")
    args = parser.parse_args()
    if not 1 <= args.limit <= 15:
        parser.error("limit must be between 1 and 15")
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", "curl/8.5.0")]
    urllib.request.install_opener(opener)
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    benchmark = (source_root() / "scripts/retarget/50_motion_benchmark.sh").read_text()
    clips = [line.split()[1] for line in benchmark.splitlines() if line.startswith("oakink2 scene_")][:args.limit]
    files = ["object_repair.tar", "program.tar"] + [f"anno_preview/{c.rsplit('__st', 1)[0]}.pkl" for c in clips]
    repo = "kelvin34501/OakInk-v2"  # Official toolkit script/download.py points here.
    with urllib.request.urlopen(f"{endpoint}/api/datasets/{repo}", timeout=60) as response:
        revision = json.load(response)["sha"]

    metadata = {}
    if args.all:
        with urllib.request.urlopen(f"{endpoint}/api/datasets/{repo}/tree/{revision}?limit=1000", timeout=60) as response:
            metadata.update({e["path"]: e for e in json.load(response) if e["type"] == "file"})
        files = ["object_repair.tar", "program.tar"]
        url = f"{endpoint}/api/datasets/{repo}/tree/{revision}/anno_preview?limit=1000"
        while url:
            url = url.replace("https://huggingface.co", endpoint)
            with urllib.request.urlopen(url, timeout=60) as response:
                entries = json.load(response)
                link = response.headers.get("Link", "")
            metadata.update({e["path"]: e for e in entries if e["type"] == "file"})
            files.extend(v["path"] for v in entries if v["type"] == "file" and v["path"].endswith(".pkl"))
            url = next((part.split("<", 1)[1].split(">", 1)[0] for part in link.split(",") if 'rel="next"' in part), None)
        clips = []
        print(f"Downloading all {len(files) - 2} recordings at {revision}", flush=True)

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args):
            return None

    def fetch(name):
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"{endpoint}/datasets/{repo}/resolve/{revision}/{urllib.parse.quote(name)}?download=true"
        tmp = target.with_suffix(target.suffix + ".partial")
        for attempt in range(10):
            try:
                if name in metadata and metadata[name].get("lfs"):
                    expected = metadata[name]["size"]
                    expected_sha = metadata[name]["lfs"]["oid"]
                else:
                    try:
                        meta = urllib.request.build_opener(NoRedirect()).open(
                            urllib.request.Request(url, method="HEAD", headers={"User-Agent": "curl/8.5.0"}), timeout=60)
                    except urllib.error.HTTPError as error:
                        if error.code not in (301, 302, 303, 307, 308):
                            raise
                        meta = error
                    with meta:
                        expected = int(meta.headers.get("X-Linked-Size") or meta.headers["Content-Length"])
                        expected_sha = (meta.headers.get("X-Linked-ETag") or "").strip('"')
                def record(path):
                    with path.open("rb") as stream:
                        digest = hashlib.file_digest(stream, "sha256").hexdigest()
                    if path.stat().st_size != expected or (len(expected_sha) == 64 and digest != expected_sha):
                        raise ValueError(f"Size/hash mismatch: {name}")
                    return {"path": name, "bytes": expected, "sha256": digest}
                if target.is_file():
                    try:
                        result = record(target)
                        print(f"Verified cached {name}", flush=True)
                        return result
                    except ValueError:
                        pass
                with urllib.request.urlopen(url, timeout=60) as response, tmp.open("wb") as output:
                    shutil.copyfileobj(response, output)
                result = record(tmp)
                tmp.replace(target)
                print(f"Downloaded {name} ({expected:,} bytes)", flush=True)
                return result
            except (OSError, ValueError) as error:
                if attempt == 9:
                    raise
                print(f"Retry {attempt + 1}/9 for {name}: {error}", flush=True)
                delay = int(error.headers.get("Retry-After", "60")) if isinstance(error, urllib.error.HTTPError) and error.code == 429 else min(30, 2 ** attempt)
                time.sleep(min(300, max(1, delay)))

    with ThreadPoolExecutor(max_workers=3) as pool:
        records = []
        for record in pool.map(fetch, files):
            records.append(record)
            (root / "download-progress.json").write_text(json.dumps({"revision": revision, "total": len(files), "verified": len(records), "files": records}, indent=2) + "\n")
    for name in ("object_repair.tar", "program.tar"):
        with tarfile.open(root / name) as archive:
            archive.extractall(root, filter="data")
    manifest = {"repo": repo, "revision": revision, "upstream_revision": REVISION,
                "clips": clips, "files": records,
                "selection": "all OakInk2 annotation recordings" if args.all else "OakInk2 entries of FlashDexRetarget 50-motion benchmark, original order"}
    (root / "download-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Ready: {root}")


if __name__ == "__main__":
    main()

"""Download official TACO pose archives or HOT3D meshes, excluding image streams."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import urllib.parse
import urllib.request
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["taco", "hot3d-assets"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", "curl/8.5.0")]
    urllib.request.install_opener(opener)

    def get_json(url):
        with urllib.request.urlopen(url, timeout=60) as response:
            return json.load(response)

    repo = "leolyliu/TACO" if args.dataset == "taco" else "bop-benchmark/hot3d"
    revision = get_json(f"{endpoint}/api/datasets/{repo}")["sha"]
    folder = "" if args.dataset == "taco" else "/object_models"
    url = f"{endpoint}/api/datasets/{repo}/tree/{revision}{folder}?limit=1000"
    entries = []
    while url:
        with urllib.request.urlopen(url.replace("https://huggingface.co", endpoint), timeout=60) as response:
            entries.extend(json.load(response))
            link = response.headers.get("Link", "")
        url = next((p.split("<", 1)[1].split(">", 1)[0] for p in link.split(",") if 'rel="next"' in p), None)
    names = {"Hand_Poses.zip", "Object_Poses.zip", "Object_Models.zip"}
    selected = [e for e in entries if e["type"] == "file" and (
        e["path"] in names if args.dataset == "taco" else e["path"].endswith(".glb"))]
    if not selected or (args.dataset == "taco" and len(selected) != 3):
        raise ValueError("Expected pose archives/GLB models not found; inspect upstream layout")
    total = sum(e["size"] for e in selected)
    if shutil.disk_usage(root).free < total * (4 if args.dataset == "taco" else 1.2) + 20_000_000_000:
        raise RuntimeError("Insufficient space for downloads, preparation and 20 GB reserve")
    manifest = {"repo": repo, "revision": revision, "dataset": args.dataset,
                "expected_files": len(selected), "expected_bytes": total,
                "complete": False, "files": []}

    def save():
        temp = root / "download-manifest.json.partial"
        temp.write_text(json.dumps(manifest, indent=2) + "\n")
        temp.replace(root / "download-manifest.json")

    save()
    print(f"Downloading {len(selected)} files / {total:,} bytes from {repo}@{revision}", flush=True)

    def fetch(entry):
        name = entry["path"]
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        expected_sha = entry.get("lfs", {}).get("oid")
        if not expected_sha:
            raise ValueError(f"Missing upstream SHA256: {name}")
        def verify(path):
            if path.stat().st_size != entry["size"]:
                raise ValueError(f"Size mismatch: {name}")
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != expected_sha:
                raise ValueError(f"SHA256 mismatch: {name}")
            return {"path": name, "bytes": entry["size"], "sha256": digest}
        for attempt in range(5):
            try:
                if target.is_file():
                    result = verify(target)
                else:
                    tmp = target.with_suffix(target.suffix + ".partial")
                    url = f"{endpoint}/datasets/{repo}/resolve/{revision}/{urllib.parse.quote(name)}?download=true"
                    with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as output:
                        shutil.copyfileobj(response, output, length=1024 * 1024)
                    result = verify(tmp)
                    tmp.replace(target)
                print(f"Verified {name}: {entry['size']:,} bytes", flush=True)
                return result
            except (OSError, ValueError) as error:
                print(f"Attempt {attempt + 1}/5 {name}: {error}", flush=True)
                if attempt == 4:
                    raise
                time.sleep(min(30, 2 ** attempt))

    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(fetch, e) for e in selected]):
            manifest["files"].append(future.result())
            save()
    if args.dataset == "taco":
        for name in sorted(names):
            with zipfile.ZipFile(root / name) as archive:
                unpacked = sum(member.file_size for member in archive.infolist())
                if shutil.disk_usage(root).free < unpacked + 20_000_000_000:
                    raise RuntimeError(f"Insufficient space to extract {name}")
                for member in archive.infolist():
                    path = (root / member.filename).resolve()
                    if not path.is_relative_to(root):
                        raise ValueError(f"Unsafe archive path: {member.filename}")
                archive.extractall(root)
            print(f"Extracted {name}", flush=True)
        meshes = list(root.rglob("*_cm.obj"))
        if not meshes:
            raise ValueError("No object meshes after extraction")
        parents = {m.parent for m in meshes}
        if len(parents) != 1:
            raise ValueError("Ambiguous object mesh directory")
        alias = root / "object_models_released"
        if not alias.exists():
            alias.symlink_to(next(iter(parents)), target_is_directory=True)
        for folder in ("Hand_Poses", "Object_Poses"):
            if not (root / folder).is_dir():
                raise ValueError(f"Missing {folder}; inspect archive directory layout")
    manifest["complete"] = True
    save()
    print("Download and extraction complete", flush=True)


if __name__ == "__main__":
    main()

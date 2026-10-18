"""Fetch pinned, public full checkpoints; preserve upstream file metadata and hashes."""
import argparse
import hashlib
import json
import time
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

MODELS = {
    "qwen3vl8b": ("Qwen/Qwen3-VL-8B-Instruct", "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"),
    "spatial_interactor8b": ("kagakouko/Spatial-Interactor-Qwen3-VL-8B", "455e99af74252e6be8fa572f6b469fd1f53fb242"),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--record", type=Path, required=True)
    p.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    a = p.parse_args()
    a.record.mkdir(parents=True, exist_ok=True)
    for key in a.models:
        repo, revision = MODELS[key]
        folder = a.root.resolve() / key
        begin = time.time()
        info = HfApi().model_info(repo, revision=revision, files_metadata=True, timeout=30)
        metadata = {"repo_id": repo, "revision": revision, "gated": info.gated,
                    "local_path": str(folder), "started_unix": begin,
                    "upstream_files": [{"name": s.rfilename, "size": s.size,
                                        "lfs": s.lfs.__dict__ if s.lfs else None}
                                       for s in info.siblings]}
        record = a.record / f"{key}.json"
        record.write_text(json.dumps(metadata, indent=2))
        print("DOWNLOAD", key, revision, flush=True)
        snapshot_download(repo, revision=revision, local_dir=folder, max_workers=4,
                          allow_patterns=["*.json", "*.txt", "*.jinja", "*.safetensors", "README.md"])
        hashes = []
        for path in sorted(folder.iterdir()):
            if not path.is_file():
                continue
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
                    digest.update(chunk)
            hashes.append({"file": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()})
        metadata.update(files=hashes, status="complete", elapsed_s=time.time() - begin)
        record.write_text(json.dumps(metadata, indent=2))
        print("MODEL_FILES_READY", key, metadata["elapsed_s"], flush=True)


if __name__ == "__main__":
    main()

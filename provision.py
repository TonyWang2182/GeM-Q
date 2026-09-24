"""Download the pinned MusicEval and MuQ releases used by GeM-Q."""

import sys, json, concurrent.futures as cf, hashlib, time
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent
REV = "20977b88b95a944bf4d55545652958c62af0d35f"
MODEL_REV = "0562a57814f6f8bbd9fdea0a25921a2fce1a841a"


def fetch(url, dest, sha=None):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (
        not sha or hashlib.sha256(dest.read_bytes()).hexdigest() == sha
    ):
        return
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(8):
        try:
            offset = part.stat().st_size if part.exists() else 0
            with requests.get(
                url,
                headers={"Range": f"bytes={offset}-"} if offset else {},
                stream=True,
                timeout=(20, 90),
            ) as r:
                r.raise_for_status()
                resume = offset and r.status_code == 206
                if resume:
                    assert r.headers.get("Content-Range", "").startswith(
                        f"bytes {offset}-"
                    )
                with part.open("ab" if resume else "wb") as f:
                    for block in r.iter_content(1024 * 1024):
                        f.write(block)
            if sha:
                assert (
                    hashlib.sha256(part.read_bytes()).hexdigest() == sha
                ), f"hash mismatch: {dest}"
            part.replace(dest)
            print("FETCHED", dest.name, flush=True)
            return
        except Exception:
            if attempt == 7:
                raise
            time.sleep(2 + attempt)


def acquire_data():
    meta = requests.get(
        "https://huggingface.co/api/datasets/BAAI/MusicEval/revision/"
        + REV
        + "?blobs=true",
        timeout=45,
    ).json()
    (ROOT / "data/hf_manifest.json").write_text(json.dumps(meta, indent=2))
    entries = [
        x
        for x in meta["siblings"]
        if x["rfilename"].startswith("data/train-")
        and x["rfilename"].endswith(".parquet")
    ]
    assert len(entries) == 3, entries
    with cf.ThreadPoolExecutor(max_workers=3) as pool:
        jobs = [
            pool.submit(
                fetch,
                f"https://huggingface.co/datasets/BAAI/MusicEval/resolve/{REV}/{e['rfilename']}",
                ROOT / "data/train.parquet" / Path(e["rfilename"]).name,
                e.get("lfs", {}).get("sha256"),
            )
            for e in entries
        ]
        for j in jobs:
            j.result()
    for split in ["validation", "test"]:
        dest = ROOT / f"data/{split}.parquet"
        entry = next(
            x
            for x in meta["siblings"]
            if x["rfilename"] == f"data/{split}-00000-of-00001.parquet"
        )
        fetch(
            f"https://huggingface.co/datasets/BAAI/MusicEval/resolve/{REV}/{entry['rfilename']}",
            dest,
            entry.get("lfs", {}).get("sha256"),
        )
    from remotezip import RemoteZip

    with RemoteZip(
        f"https://huggingface.co/datasets/BAAI/MusicEval/resolve/{REV}/MusicEval-full.zip"
    ) as z:
        for n in z.namelist():
            if Path(n).name in [
                "prompt_info.txt",
                "demo_prompt_info.txt",
                "system_info_EN.xlsx",
                "README.md",
            ]:
                (ROOT / "data" / Path(n).name).write_bytes(z.read(n))


def acquire_model():
    repo = "OpenMuQ/MuQ-large-msd-iter"
    res = requests.get(
        f"https://huggingface.co/api/models/{repo}/revision/{MODEL_REV}?blobs=true",
        timeout=45,
    )
    res.raise_for_status()
    meta = res.json()
    (ROOT / "models/muq_manifest.json").write_text(json.dumps(meta, indent=2))
    entries = [
        e
        for e in meta["siblings"]
        if e["rfilename"].endswith((".json", ".safetensors", ".bin"))
    ]
    with cf.ThreadPoolExecutor(max_workers=3) as pool:
        jobs = [
            pool.submit(
                fetch,
                f"https://huggingface.co/{repo}/resolve/{meta['sha']}/{e['rfilename']}",
                ROOT / "models/muq" / e["rfilename"],
                e.get("lfs", {}).get("sha256"),
            )
            for e in entries
        ]
        for j in jobs:
            j.result()


if __name__ == "__main__":
    for name in ["data", "models", "features", "results"]:
        (ROOT / name).mkdir(exist_ok=True)
    with cf.ThreadPoolExecutor(max_workers=3) as pool:
        for f in [pool.submit(x) for x in [acquire_data, acquire_model]]:
            f.result()
    print("PROVISION_COMPLETE", flush=True)

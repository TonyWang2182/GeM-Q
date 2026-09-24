"""Extract immutable center-crop MuQ frame caches with input/data integrity audit."""

import os, sys, json, hashlib, io, re, math, time, csv
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import resample_poly
import torch
from muq import MuQ

ROOT = Path(__file__).resolve().parent


def main():
    torch.set_num_threads(6)
    torch.manual_seed(20260924)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    prompts = {}
    for p in ["prompt_info.txt", "demo_prompt_info.txt"]:
        with (ROOT / "data" / p).open(encoding="utf-8-sig") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                prompts[row["id"].strip()] = row["text"].strip()
    model = MuQ.from_pretrained(str(ROOT / "models/muq")).eval().cuda()
    print("MODEL_LOADED", sum(p.numel() for p in model.parameters()), flush=True)
    records = []
    hashes = {}
    start = time.time()
    (ROOT / "features").mkdir(exist_ok=True)
    with torch.inference_mode():
        for split in ["train", "validation", "test"]:
            rows = pq.read_table(ROOT / f"data/{split}.parquet").to_pylist()
            for j, r in enumerate(rows):
                a = r["audio"]
                name = Path(a["path"]).name
                system, pid = re.search(r"(S\d+)_(P\d+|D\d+)", name).groups()
                if system in {"S001", "S006", "S013", "S017", "S018", "S033"}:
                    assert name in prompts, name
                prompt = prompts.get(name, prompts.get(pid))
                assert prompt is not None, name
                x, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32", always_2d=True)
                pcmhash = hashlib.sha256(x.tobytes() + str(sr).encode()).hexdigest()
                if pcmhash in hashes:
                    raise ValueError(
                        f"Duplicate PCM {name} {hashes[pcmhash]}: resolve before training"
                    )
                hashes[pcmhash] = name
                duration = len(x) / sr
                x = x.mean(1)
                if sr != 24000:
                    g = math.gcd(sr, 24000)
                    x = resample_poly(x, 24000 // g, sr // g).astype(np.float32)
                offset = max(0, (len(x) - 240000) // 2)
                x = x[offset : offset + 240000]
                x = np.pad(x, (0, 240000 - len(x)))
                dest = ROOT / "features" / f"{Path(name).stem}.npy"
                if not dest.exists():
                    out = model(
                        torch.from_numpy(x).unsqueeze(0).cuda(),
                        output_hidden_states=False,
                    )
                    h = (
                        out.last_hidden_state[0]
                        .float()
                        .cpu()
                        .numpy()
                        .astype(np.float16)
                    )
                    assert (
                        h.ndim == 2 and h.shape[1] == 1024 and np.isfinite(h).all()
                    ), h.shape
                    np.save(dest, h)
                else:
                    h = np.load(dest, mmap_mode="r")
                records.append(
                    dict(
                        id=name,
                        system=system,
                        prompt=prompt,
                        prompt_hash=hashlib.sha256(prompt.encode()).hexdigest(),
                        original_split=split,
                        quality=float(r["overall quality"]),
                        alignment=float(r["textual alignment"]),
                        duration=duration,
                        sample_rate=sr,
                        offset_24k=offset,
                        file_sha256=hashlib.sha256(a["bytes"]).hexdigest(),
                        pcm_sha256=pcmhash,
                        feature=str(dest.relative_to(ROOT)),
                        shape=list(h.shape),
                    )
                )
                if j % 25 == 0:
                    print(
                        "EXTRACT",
                        split,
                        j,
                        len(rows),
                        "elapsed",
                        round(time.time() - start, 1),
                        flush=True,
                    )
            del rows
    assert len(records) == 2748, len(records)
    assert len({r["id"] for r in records}) == len(records)
    (ROOT / "data/records.json").write_text(json.dumps(records, indent=2))
    manifest = dict(
        n=len(records),
        systems=sorted({r["system"] for r in records}),
        seconds=time.time() - start,
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(),
        precision="fp32 inference; fp16 frame storage",
        crop="center 10s; zero pad",
        encoder_mode="eval",
        target="overall quality",
    )
    (ROOT / "features/manifest.json").write_text(json.dumps(manifest, indent=2))
    print("EXTRACTION_COMPLETE", manifest, flush=True)


if __name__ == "__main__":
    main()

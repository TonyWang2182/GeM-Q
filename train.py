"""Matched frozen-MuQ quality heads with nested generator-disjoint evaluation."""

import argparse, copy, hashlib, json, random, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.stats import spearmanr, pearsonr

ROOT = Path(__file__).resolve().parent


class Head(nn.Module):
    def __init__(self, dim=1024):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, 128), nn.Tanh(), nn.Linear(128, 1)
        )
        self.mlp = nn.Sequential(
            nn.Linear(dim, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 1)
        )

    def forward(self, x):
        a = self.attention(x).squeeze(-1).softmax(-1)
        return self.mlp(torch.bmm(a.unsqueeze(1), x).squeeze(1)).squeeze(-1)


def make_splits(records):
    g = np.array([r["system"] for r in records])
    systems = sorted(set(g))
    shuffled = np.random.default_rng(20260924).permutation(systems)
    folds = [list(x) for x in np.array_split(shuffled, 5)]
    splits = []
    for k in range(5):
        test = np.where(np.isin(g, folds[k]))[0]
        val = np.where(np.isin(g, folds[(k + 1) % 5]))[0]
        tr = np.where(~np.isin(g, folds[k] + folds[(k + 1) % 5]))[0]
        splits.append(
            dict(
                name=f"ood{k}", train=tr.tolist(), val=val.tolist(), test=test.tolist()
            )
        )
    for seed in [42, 43, 44]:
        rng = np.random.default_rng(seed)
        parts = [[], [], []]
        for s in systems:
            ids = rng.permutation(np.where(g == s)[0])
            n = len(ids)
            a = int(0.6 * n)
            b = int(0.8 * n)
            for p, idx in zip(parts, [ids[:a], ids[a:b], ids[b:]]):
                p.extend(idx.tolist())
        splits.append(
            dict(name=f"iid{seed}", train=parts[0], val=parts[1], test=parts[2])
        )
    for sp in splits:
        assert (
            len(set(sp["train"]) & set(sp["val"]))
            == len(set(sp["train"]) & set(sp["test"]))
            == 0
        )
        assert len(sp["train"]) + len(sp["val"]) + len(sp["test"]) == len(records)
        if sp["name"].startswith("ood"):
            assert not (set(g[sp["train"]]) & set(g[sp["test"]]))
            assert not (set(g[sp["train"]]) & set(g[sp["val"]]))
    alltest = [i for sp in splits[:5] for i in sp["test"]]
    assert sorted(alltest) == list(range(len(records)))
    return splits


def macro(values, groups):
    return float(np.mean([np.mean(values[groups == g]) for g in np.unique(groups)]))


def evaluate(model, x, ids):
    model.eval()
    out = []
    with torch.inference_mode():
        for b in range(0, len(ids), 64):
            out.append(model(x[ids[b : b + 64]].float()).cpu().numpy())
    return np.concatenate(out)


def fit(x, y, groups, split, seed, method, lam=0.5, eta=0.1):
    label = f"{split['name']}_{seed}_{method}"
    dest = ROOT / "results" / label
    dest.mkdir(exist_ok=True)
    if (dest / "done.json").exists():
        print("REUSE_RUN", label, flush=True)
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = Head().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    tr = np.array(split["train"])
    va = np.array(split["val"])
    te = np.array(split["test"])
    tg = groups[tr]
    ug = np.unique(tg)
    G = len(ug)
    counts = np.array([(tg == g).sum() for g in ug])
    q = np.ones(G) / G
    yi = torch.tensor(y, device="cuda", dtype=torch.float32)
    best = float("inf")
    best_state = None
    bad = 0
    history = []
    rng = np.random.default_rng(seed)
    tick = time.time()
    for epoch in range(30):
        mix = np.ones(G) / G if method == "gb" else (1 - lam) / G + lam * q
        sample = np.ones(len(tr))
        if method != "erm":
            for j, g in enumerate(ug):
                sample[tg == g] = len(tr) * mix[j] / counts[j]
        assert np.isclose(sample.mean(), 1), sample.mean()
        perm = rng.permutation(len(tr))
        model.train()
        loss_sum = 0
        for b in range(0, len(perm), 32):
            pos = perm[b : b + 32]
            ids = tr[pos]
            optimizer.zero_grad(set_to_none=True)
            pred = model(x[ids].float())
            weights = torch.tensor(sample[pos], device="cuda", dtype=torch.float32)
            loss = ((pred - yi[ids]).square() * weights).mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite loss {label}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(pos)
        vp = evaluate(model, x, va)
        v_loss = macro((vp - y[va]) ** 2, groups[va])
        tp = evaluate(model, x, tr)
        gl = np.array([np.mean((tp[tg == g] - y[tr][tg == g]) ** 2) for g in ug])
        history.append(
            dict(
                epoch=epoch,
                train_weighted_mse=loss_sum / len(tr),
                val_macro_mse=v_loss,
                group_loss=gl.tolist(),
                group_weights=mix.tolist(),
                groups=ug.tolist(),
            )
        )
        if method not in ["erm", "gb"]:
            logq = np.log(q) + eta * gl
            logq -= logq.max()
            q = np.exp(logq)
            q /= q.sum()
        if v_loss < best - 1e-8:
            best = v_loss
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            best_epoch = epoch
            bad = 0
        else:
            bad += 1
        if bad >= 7:
            break
    model.load_state_dict(best_state)
    # Test labels used only after the validation-selected checkpoint is frozen.
    pred = evaluate(model, x, te) * 4 + 1
    target = y[te] * 4 + 1
    rows = [
        dict(index=int(i), prediction=float(p), target=float(t), system=str(groups[i]))
        for i, p, t in zip(te, pred, target)
    ]
    (dest / "predictions.json").write_text(json.dumps(rows))
    (dest / "history.json").write_text(json.dumps(history))
    torch.save(best_state, dest / "head.pt")
    report = dict(
        split=split["name"],
        seed=seed,
        method=method,
        lambda_value=lam,
        eta=eta,
        best_epoch=best_epoch,
        epochs=len(history),
        seconds=time.time() - tick,
        n_train=len(tr),
        n_val=len(va),
        n_test=len(te),
        train_groups=ug.tolist(),
        val_groups=np.unique(groups[va]).tolist(),
        test_groups=np.unique(groups[te]).tolist(),
        mae=float(np.mean(abs(pred - target))),
        macro_mae=macro(abs(pred - target), groups[te]),
        srcc=float(spearmanr(pred, target).statistic),
        pcc=float(pearsonr(pred, target).statistic),
        head_parameters=sum(p.numel() for p in model.parameters()),
    )
    (dest / "done.json").write_text(json.dumps(report, indent=2))
    print("RUN_DONE", report, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["main", "sensitivity", "all"], default="all")
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    records = json.loads((ROOT / "data/records.json").read_text())
    splits = make_splits(records)
    (ROOT / "data/splits.json").write_text(json.dumps(splits, indent=2))
    arrays = [np.load(ROOT / r["feature"]) for r in records]
    assert len({a.shape for a in arrays}) == 1
    x = torch.from_numpy(np.stack(arrays)).cuda()
    del arrays
    assert torch.isfinite(x).all()
    y = (np.array([r["quality"] for r in records]) - 1) / 4
    groups = np.array([r["system"] for r in records])
    print(
        "CACHE_READY",
        list(x.shape),
        "GB",
        x.numel() * x.element_size() / 1e9,
        flush=True,
    )
    if args.suite in ["main", "all"]:
        for split in splits:
            seeds = (
                [42, 43, 44]
                if split["name"].startswith("ood")
                else [int(split["name"][3:])]
            )
            for seed in seeds:
                for method in ["erm", "gb", "sgb"]:
                    fit(x, y, groups, split, seed, method)
    if args.suite in ["sensitivity", "all"]:
        for split in splits[:5]:
            for method, lam, eta in [
                ("dro", 1.0, 0.1),
                ("lambda025", 0.25, 0.1),
                ("eta005", 0.5, 0.05),
                ("eta020", 0.5, 0.2),
            ]:
                fit(x, y, groups, split, 42, method, lam, eta)
    (ROOT / "results/completed.json").write_text(
        json.dumps(dict(suite=args.suite, completed=time.time()))
    )
    print("TRAINING_COMPLETE", flush=True)


if __name__ == "__main__":
    main()

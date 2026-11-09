"""Run one optimizer step per objective on a small feature tensor."""

import torch

from model import Head
from objectives import (
    erm_mse,
    gb_weights,
    sgb_weights,
    training_counts,
    update_q,
    weighted_mse,
)


def main():
    torch.manual_seed(42)
    features = torch.randn(6, 5, 8)
    mos = torch.linspace(2.0, 4.5, 6)
    targets = (mos - 1.0) / 4.0
    groups = torch.tensor([0, 1, 1, 2, 2, 2], dtype=torch.long)
    counts = training_counts(groups)
    print(f"Features: {tuple(features.shape)}")

    for method in ("erm", "gb", "sgb"):
        torch.manual_seed(42)
        model = Head(dim=features.shape[-1])
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        q = torch.full((len(counts),), 1.0 / len(counts), dtype=torch.float64)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(features)
        if method == "erm":
            loss = erm_mse(prediction, targets)
        else:
            weights = gb_weights(counts) if method == "gb" else sgb_weights(counts, q)
            loss = weighted_mse(prediction, targets, weights[groups])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if method == "sgb":
            model.eval()
            with torch.no_grad():
                errors = (model(features).double() - targets.double()).square()
                group_losses = torch.stack(
                    [errors[groups == g].mean() for g in range(len(counts))]
                )
            q = update_q(q, group_losses)
            assert torch.isfinite(q).all() and torch.isclose(q.sum(), q.new_tensor(1.0))

        print(f"{method.upper()}: predictions {tuple(prediction.shape)}, update complete")


if __name__ == "__main__":
    main()

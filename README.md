# GeM-Q

Core PyTorch modules for generator-balanced music quality assessment.

## Modules

| File | Contents |
| --- | --- |
| `model.py` | Attention pooling and an MLP quality head |
| `objectives.py` | ERM, GB, SGB, and the adaptive weight update |
| `example.py` | One optimizer step per objective |

## Install

Use Python 3.10 or later.

```bash
git clone https://github.com/TonyWang2182/GeM-Q.git
cd GeM-Q
python -m pip install -r requirements.txt
```

## Usage

```bash
python example.py
```

The example uses a small feature tensor. For cached MuQ features, pass a
float32 tensor of shape `(batch, frames, 1024)` to `Head()`. The output has
shape `(batch,)`. Train against `(MOS - 1) / 4` and recover MOS with
`1 + 4 * prediction`.

Generator IDs are contiguous integers from `0` to `G - 1`. Call
`training_counts` on the full training partition. The weight functions return
one per-clip weight per generator; index these with the batch's generator IDs.
`weighted_mse` divides by batch size.

For SGB, initialize `q` uniformly in float64 and hold its mixture weights fixed
within each epoch. After the epoch, recompute each training generator's mean
squared error in evaluation mode and pass those losses to `update_q`.
Compute these group losses in float64.

## Sources

The head follows the A1 implementation in
[MuQ-Eval](https://github.com/dgtql/MuQ-Eval).
[MuQ](https://github.com/tencent-ailab/MuQ) provides the frame encoder.
The adaptive update follows
[GroupDRO](https://github.com/kohpangwei/group_DRO).
The MuQ-Eval license is retained in [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES).

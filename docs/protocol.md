# Experimental protocol

## Data and representation

MusicEval contains 2,748 clips from 31 generators. Overall-quality MOS is the sole target. The original partitions are pooled to create new generator-disjoint folds. This is an exploratory study, not a preregistration or an official challenge submission.

| Component | Setting |
|:--|:--|
| Dataset revision | `20977b88b95a944bf4d55545652958c62af0d35f` |
| MuQ model | `OpenMuQ/MuQ-large-msd-iter` |
| Model revision | `0562a57814f6f8bbd9fdea0a25921a2fce1a841a` |
| MuQ-Eval architecture reference | `60a88f8ac0909ca1fd1a788af3660f1fc376977a1` |
| Input | Mono, 24 kHz, center 10 seconds, zero-pad short clips |
| Frames | Last hidden state, 250 × 1024 |
| Precision | FP32 encoder inference, FP16 cache, FP32 head |
| Target | `(MOS - 1) / 4`; invert the transform without clipping |

System identifiers are parsed from filenames. Extraction checks file hashes, decoded PCM hashes, and duplicate identities. The model is frozen in evaluation mode; no fitted preprocessing uses the held-out labels.

## Head and optimization

Attention uses `Linear(1024, 128)`, tanh, `Linear(128, 1)`, and a temporal softmax. A weighted sum pools the frames. The regressor uses `Linear(1024, 256)`, GELU, dropout 0.1, and `Linear(256, 1)`. There are 393,986 trainable parameters.

- AdamW: learning rate 0.001, weight decay 0.01.
- Batch size 32, gradient norm cap 1, at most 30 epochs.
- Validation generator-macro MSE selects the checkpoint, patience seven epochs.
- Seeds 42, 43, and 44, paired initializations and clip order across objectives.
- ERM: unweighted clip-average MSE.
- GB / GeM-Q: average of generator-mean MSEs.
- SGB: `(1-lambda)/G + lambda*q_g`, with `lambda=0.5`, `eta=0.1`.
- Initialize `q` uniformly. Update it once per epoch using detached full-training-generator losses in model evaluation mode: `q_g <- q_g * exp(eta * L_g)`, then normalize.

Per-clip weights are `N*w_g/n_g`. Their mean over the training partition is one. Divide minibatch loss by batch size. Fixed sensitivity runs change one setting: `lambda=0.25`, `lambda=1`, `eta=0.05`, or `eta=0.2`, all with seed 42.

## Partitions

Sort generator IDs, shuffle with seed 20260924, and split into five groups. Outer fold `k` uses group `k` for test, `(k+1)%5` for validation, and the other three for training. Each clip has one held-out prediction per seed. A separate seen-generator control uses per-generator 60/20/20 clip splits.

Twenty-five systems have 100 clips. The other counts are 7, 8, 21, 25, 54, and 133. In zero-indexed outer fold 3, every training generator has 100 clips, making GB and ERM identical. The original saved predictions match exactly for all three seeds in this fold.

The five test folds share 100/100, 100/100, 100/100, 100/160, and 100/200 distinct prompt texts with training. Related generator families may cross folds. This evaluation does not establish unseen-prompt or unseen-family transfer.

## Statistics and reported evidence

The primary metric is generator-macro MAE on the original MOS scale. Compute metrics from each seed's concatenated outer-test predictions and average metrics across seeds, rather than ensembling predictions. Pooled SRCC and mean within-generator SRCC are separate quantities. The latter excludes groups with fewer than five clips or constant targets/predictions. Tail summaries are the 75th percentile and maximum of generator MAEs.

Paired uncertainty uses 5,000 generator-cluster resamples, seed 20260924, on per-generator errors averaged across the fitted seeds. These intervals condition on the fitted models; they are not independent-dataset or independent-family confidence statements.

The original 74-run audit checked partitions, saved prediction labels, validation-selected epochs, and repeated inference. Raw checkpoints, training histories, audio, prompts, and environment-specific files are intentionally excluded from this compact repository. `evaluate.py` recomputes numerical summaries after a full local run.

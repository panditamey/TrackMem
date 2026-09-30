# TrackMem

Research code for a lightweight persistent-memory tiny-object tracker (shuttlecock), trained from scratch.

Models (`--model`):
- `trackmem`: TrackNetV5 + a learned persistent kinematic memory (position, velocity, acceleration,
  uncertainty, latent state) fed back into detection as prior heatmaps, with visibility and sub-pixel
  offset heads. Trained recurrently on 8-frame sequences; inference carries the memory across whole
  rallies.
- `tracknetv5`: baseline. Re-implementation of [TrackNetV5](https://arxiv.org/abs/2512.02789), verified
  output-equivalent to the [official code](https://github.com/thaonan/TrackNetV5-SDK) (14.77M params).
  Trained with our own recipe and evaluated on the official held-out Test matches, so numbers are not
  directly comparable to the paper.

## Data

Shuttlecock Trajectory Dataset (TrackNetV2), unzipped to `data/TracknetV2/{Professional,Amateur,Test}`.
Decode videos once into 512x288 frame arrays (~40 GB):

```bash
python tools/extract_frames.py --root data/TracknetV2 --out data/cache/frames_512x288
```

## Train / evaluate

```bash
python train.py --model trackmem                          # any setting: --set train.batch_size=4 ...
python train.py --model tracknetv5
python train.py --model trackmem --resume runs/trackmem/last.pt
python evaluate.py --ckpt runs/trackmem/best.pt --split test
python evaluate.py --ckpt runs/trackmem/best.pt --split val --set eval.memory_mode=reset   # memory-use test
python tools/kalman_baseline.py --frames runs/tracknetv5/eval_val/frames.csv            # open-loop control
```

Evaluation reports TP/FP1/FP2/TN/FN metrics at 4 px tolerance in both 1280x720 and 512x288 space,
overall and per subset (normal, fast, hit, reappear, short/long occlusion, out-of-frame), and writes
per-frame predictions to `eval_<split>/frames.csv`.

## Annotated video

```bash
python inference.py --ckpt runs/trackmem/best.pt       # first Test rally video
python inference.py --video match.mp4 --out outputs/match_pred.mp4
```
Writes the annotated video (prediction red + trail, TrackMem memory estimate yellow while undetected,
GT green when a dataset CSV exists) and a CSV of predictions in the dataset label format
(`Frame,Visibility,X,Y,Peak`, plus `VisProb,LatentX,LatentY` for TrackMem).

## Colab

Open in Colab: https://colab.research.google.com/github/panditamey/TrackMem/blob/main/colab/train.ipynb
Needs `MyDrive/TrackNetDataset/TrackNetV2.zip` in Drive. Set `MODEL` in the first cell.

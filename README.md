# TrackNetV6

Research code for a lightweight persistent-memory tiny-object tracker (shuttlecock), trained from scratch.
Current stage: **B0**, a TrackNetV5-like baseline (TrackNetV2 U-Net + MDD motion input, WBCE loss; no R-STR, no memory).

## Data

Shuttlecock Trajectory Dataset (TrackNetV2), unzipped to `data/TracknetV2/{Professional,Amateur,Test}`.
Decode videos once into 512x288 frame arrays (~40 GB):

```bash
python tools/extract_frames.py --root data/TracknetV2 --out data/cache/frames_512x288
```

## Train / evaluate

```bash
python train.py --config config.yaml                      # any setting: --set train.batch_size=4 ...
python train.py --config config.yaml --resume runs/b0_v5like/last.pt
python evaluate.py --ckpt runs/b0_v5like/best.pt --split test
```

Evaluation reports TP/FP1/FP2/TN/FN metrics at 4 px tolerance in both 1280x720 and 512x288 space,
overall and per subset (normal, fast, hit, reappear, short/long occlusion, out-of-frame), and writes
per-frame predictions to `eval_<split>/frames.csv`.

## Annotated video

```bash
python inference.py                                   # best ckpt, first Test rally video
python inference.py --video match.mp4 --out outputs/match_pred.mp4
```
Writes the annotated video (prediction red + trail, GT green when a dataset CSV exists) and a CSV of
predictions in the dataset label format (`Frame,Visibility,X,Y,Peak`).

## Colab

Open in Colab: https://colab.research.google.com/github/panditamey/TrackNetV6/blob/main/colab/train_b0.ipynb Needs `MyDrive/TrackNetDataset/TrackNetV2.zip` in Drive.

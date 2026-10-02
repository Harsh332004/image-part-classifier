# Machine Part Classifier

Classifies images of machine parts (bolt, nut, washer, locating pin) with an
**EfficientNetB0** network fine-tuned end-to-end (transfer learning), served by a
Streamlit app.

## Setup (Python 3.10 or 3.11)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-train.txt
```

## Train

```bash
python check_dataset.py "C:/Users/hp340/OneDrive/Desktop/archive/dataset"
python train.py --data_dir "C:/Users/hp340/OneDrive/Desktop/archive/dataset"
```

The dataset's own `training/` and `testing/` folders contain the same parts (and
some byte-identical files), so `train.py` merges them and re-splits **by part**
so no part appears in more than one of train / validation / test.

Outputs go to `model/`: `part_classifier.keras`, `labels.json`, `metrics.json`,
`confusion_matrix.png`, `training_curves.png`.

## Run

```bash
streamlit run app.py          # web app
python predict.py photo.jpg   # command line
```

## What changed vs. the first version

| Problem | Fix |
|---|---|
| Training used RGB, the app used OpenCV BGR (silent channel swap) | One shared `common.py` preprocessing used everywhere |
| Frozen features + Random Forest | Two-phase fine-tuning of EfficientNetB0 + softmax head |
| No augmentation | Flip, rotation, zoom, shift, contrast, brightness |
| Possible train/test leakage | Exact-duplicate removal + stratified train/val/test split |
| Class imbalance ignored | Balanced class weights, label smoothing |
| Single accuracy number | Per-class report, confusion matrix, most-confident mistakes |
| Forced guess on any image | Confidence threshold -> "uncertain" |
| Model reloaded each interaction | `st.cache_resource` |
| Transparent PNGs became black | Flattened onto white |
| Pickle tied to sklearn version | Portable `.keras` model |

## Limits (be aware)

The model only knows the 4 training classes and has no "none of these" class.
The confidence threshold reduces, but does not eliminate, wrong answers on
unrelated images. Accuracy on your own real-world photos can be lower than test
accuracy on the dataset; add a few of your own photos per class to the dataset
and retrain for the best results.

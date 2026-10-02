"""Command-line prediction:  python predict.py img1.jpg img2.png [--top 3]"""
import argparse

from common import load_artifacts, load_rgb, predict_proba

parser = argparse.ArgumentParser()
parser.add_argument("images", nargs="+")
parser.add_argument("--top", type=int, default=3)
parser.add_argument("--no-tta", action="store_true")
args = parser.parse_args()

model, classes = load_artifacts()
for path in args.images:
    probs = predict_proba(model, load_rgb(path), tta=not args.no_tta)
    ranked = probs.argsort()[::-1][: args.top]
    print(path)
    for i in ranked:
        print(f"   {classes[i]:<15} {probs[i]:.1%}")

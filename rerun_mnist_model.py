"""Tier-0 replication of RBT4DNN RQ4/RQ5 pass rates for MNIST.

Runs the paper's exact LC (farleyknight-org-username/vit-base-mnist) on the
100 published LoRA-generated images per requirement in the artifact
(data/images_from_loras/{M1..M7}), mirroring rq4_mnist.py from
github.com/less-lab-uva/RBT4DNN (same processor/model/argmax oracle).

Reference values (artifact results/mnist_r*_passrate.txt, iteration 9,
1000 fresh generations x 10 reps): M1 .999 M2 .977 M3 .724 M4 .982
M5 .994 M6 .976 M7 .981. Our run uses the 100 published images per
requirement, so expect agreement within sampling error (n=100).

Usage: python rerun_mnist_model.py [--batch 25]
"""
import argparse, glob, os
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

TARGET = {'M1': 2, 'M2': 3, 'M3': 7, 'M4': 9, 'M5': 6, 'M6': 0, 'M7': 8}
PAPER = {'M1': .999, 'M2': .977, 'M3': .724, 'M4': .982, 'M5': .994, 'M6': .976, 'M7': .981}
BASE = os.path.join(os.path.dirname(__file__), 'rbt4dnn-artifact', 'mnist-images')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', type=int, default=25)
    ap.add_argument('--variants', nargs='*', default=['', 'Allreq_', 'Alldata_'])
    args = ap.parse_args()

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    processor = AutoImageProcessor.from_pretrained("farleyknight-org-username/vit-base-mnist")
    model = AutoModelForImageClassification.from_pretrained(
        "farleyknight-org-username/vit-base-mnist").to(device).eval()
    print(f"device={device}", flush=True)
    print(f"{'req':12s} {'n':>4s} {'pass%':>7s} {'paper':>6s}  fails(pred)", flush=True)

    for variant in args.variants:
        for req, target in TARGET.items():
            d = os.path.join(BASE, variant + req)
            files = sorted(glob.glob(d + '/*.png'))
            if not files:
                continue
            preds = []
            for i in range(0, len(files), args.batch):
                imgs = [Image.open(f).convert('RGB') for f in files[i:i + args.batch]]
                with torch.no_grad():
                    inputs = processor(imgs, return_tensors='pt').to(device)
                    preds += model(**inputs).logits.argmax(-1).tolist()
            fails = [(os.path.basename(files[i]), p) for i, p in enumerate(preds) if p != target]
            ref = f"{PAPER[req]:.3f}" if variant == '' else '—'
            rate = 100 * (len(files) - len(fails)) / len(files)
            print(f"{variant + req:12s} {len(files):>4d} {rate:>6.1f}% {ref:>6s}  {fails[:6]}", flush=True)


if __name__ == '__main__':
    main()

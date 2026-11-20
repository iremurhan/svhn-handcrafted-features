"""Compare hand-crafted features and classical classifiers on SVHN.

    python main.py --train data/train_32x32.mat --test data/test_32x32.mat \
        [--features hog hog_fine lbp ...] [--models knn random_forest ...] \
        [--train-size 20000] [--out results.csv]
"""
import argparse
import time

import pandas as pd

from svhn_features.data import load_svhn, subsample
from svhn_features.evaluation import fisher_ratio, fit_and_score, make_models
from svhn_features.features import EXTRACTORS


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--features", nargs="+", default=list(EXTRACTORS), choices=list(EXTRACTORS))
    parser.add_argument("--models", nargs="+", default=list(make_models()),
                        choices=list(make_models()))
    parser.add_argument("--train-size", type=int, help="stratified training subset (default: all)")
    parser.add_argument("--test-size", type=int, help="stratified test subset (default: all)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results.csv")
    args = parser.parse_args()

    X_train_img, y_train = subsample(*load_svhn(args.train), args.train_size, args.seed)
    X_test_img, y_test = subsample(*load_svhn(args.test), args.test_size, args.seed)
    print(f"train {len(y_train)} / test {len(y_test)} images")

    rows = []
    for feat in args.features:
        t0 = time.time()
        F_train, F_test = EXTRACTORS[feat](X_train_img), EXTRACTORS[feat](X_test_img)
        ratio = fisher_ratio(F_train, y_train)
        print(f"\n{feat}: dim={F_train.shape[1]}, Fisher ratio={ratio:.4f} "
              f"({time.time() - t0:.0f}s to extract)")
        for name in args.models:
            model = make_models(args.seed)[name]
            t0 = time.time()
            scores = fit_and_score(model, F_train, y_train, F_test, y_test)
            rows.append({"feature": feat, "dim": F_train.shape[1], "fisher_ratio": ratio,
                         "model": name, **scores, "seconds": time.time() - t0})
            print(f"  {name:18s} train {scores['train_accuracy']:.3f}  "
                  f"test {scores['test_accuracy']:.3f}")

    table = pd.DataFrame(rows)
    table.to_csv(args.out, index=False)
    print("\nTest accuracy (rows: feature, columns: model)")
    print(table.pivot(index="feature", columns="model", values="test_accuracy").round(3).to_string())


if __name__ == "__main__":
    main()

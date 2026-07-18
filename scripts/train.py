"""Training utilities. In production, run this file directly instead of
the notebook -- it performs the same steps non-interactively.

``cross_validate`` gives a split-independent accuracy/F1 estimate:
stratified K-fold, with augmentation applied *inside* each fold (fit on
the fold's train patterns only) so augmented variants never leak into
that fold's validation patterns.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from augment_data import augment_pattern


def cross_validate(patterns, labels, extractor, processor, n_splits=5, random_state=42):
    def preprocess(raw: str) -> str:
        return processor.preprocess(extractor.replace_with_placeholder(raw))

    patterns = np.array(patterns)
    labels = np.array(labels)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    accs, f1s = [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(patterns, labels), start=1):
        p_train, y_train_lbl = patterns[train_idx], labels[train_idx]
        p_val, y_val_lbl = patterns[val_idx], labels[val_idx]

        aug_p, aug_y = [], []
        for p, y in zip(p_train, y_train_lbl):
            for variant in augment_pattern(p, y):
                aug_p.append(variant)
                aug_y.append(y)

        X_train_text = [preprocess(p) for p in aug_p]
        X_val_text = [preprocess(p) for p in p_val]

        le = LabelEncoder().fit(list(aug_y) + list(y_val_lbl))
        y_train = le.transform(aug_y)
        y_val = le.transform(y_val_lbl)

        tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), min_df=1,
                                 max_df=0.9, token_pattern=r"\b\w+\b")
        X_train = tfidf.fit_transform(X_train_text)
        X_val = tfidf.transform(X_val_text)

        clf = SVC(kernel="linear", C=1.0, probability=True,
                  class_weight="balanced", random_state=random_state)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
        accs.append(acc)
        f1s.append(f1)
        print(f"  Fold {fold}/{n_splits}:  acc={acc*100:5.2f}%   macro-F1={f1*100:5.2f}%")

    accs, f1s = np.array(accs), np.array(f1s)
    print(f"\nCV accuracy:  {accs.mean()*100:.2f}% +/- {accs.std()*100:.2f}%")
    print(f"CV macro-F1:  {f1s.mean()*100:.2f}% +/- {f1s.std()*100:.2f}%")
    return {"accuracy_mean": accs.mean(), "accuracy_std": accs.std(),
            "f1_mean": f1s.mean(), "f1_std": f1s.std()}

"""Baseline Random Forest + SVM trên đặc trưng MFCC (mean/std), cross-validation theo quả.

- Chia fold bằng StratifiedGroupKFold với groups = fruit_id, để 3 gờ của cùng một quả
  không rơi vào cả train lẫn test (tránh rò rỉ).
- class_weight='balanced' vì lớp `im` rất ít.
- In accuracy, macro-F1, F1 từng lớp, confusion matrix (text + PNG), lưu model .joblib
  (fit lại trên toàn bộ dữ liệu sau khi CV).

Chạy:
  python -m src.train_baseline --features data/processed/features.npz --classes im,m,om --folds 5
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

MIN_FRUITS_WARN = 10


def make_models(seed: int) -> dict[str, object]:
    return {
        "rf": RandomForestClassifier(
            n_estimators=500, max_features="sqrt", class_weight="balanced_subsample", random_state=seed, n_jobs=-1
        ),
        "svm": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svc", SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced", random_state=seed)),
            ]
        ),
    }


def load_features(path: Path, classes: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Đọc features.npz; nếu `classes` là tập con của classes trong file thì lọc và đánh số lại."""
    d = np.load(path, allow_pickle=False)
    file_classes = [str(c) for c in d["classes"]]
    missing = [c for c in classes if c not in file_classes]
    if missing:
        raise SystemExit(f"Lớp {missing} không có trong {path} (có: {file_classes}). Chạy lại src.features với --classes phù hợp.")
    remap = {file_classes.index(c): i for i, c in enumerate(classes)}
    keep = np.isin(d["y"], list(remap))
    y = np.array([remap[int(v)] for v in d["y"][keep]], dtype=np.int64)
    return d["X_mfcc"][keep], y, d["groups"][keep]


def cross_validate(model, X, y, groups, n_folds: int, seed: int) -> tuple[np.ndarray, list[float]]:
    """Trả về dự đoán out-of-fold cho mọi mẫu và macro-F1 từng fold."""
    n_folds = min(n_folds, len(set(groups.tolist())))
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    pred = np.full_like(y, -1)
    fold_f1 = []
    for tr, te in cv.split(X, y, groups):
        model.fit(X[tr], y[tr])
        pred[te] = model.predict(X[te])
        fold_f1.append(f1_score(y[te], pred[te], average="macro", zero_division=0))
    return pred, fold_f1


def plot_cm(cm: np.ndarray, classes: list[str], title: str, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Thực tế")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def report(name: str, y, pred, classes: list[str], fold_f1: list[float], out_dir: Path) -> dict:
    acc = accuracy_score(y, pred)
    f1m = f1_score(y, pred, average="macro", zero_division=0)
    cm = confusion_matrix(y, pred, labels=list(range(len(classes))))
    print(f"\n===== {name.upper()} =====")
    print(f"accuracy (OOF)   : {acc:.3f}")
    print(f"macro-F1 (OOF)   : {f1m:.3f}   | theo fold: {np.mean(fold_f1):.3f} ± {np.std(fold_f1):.3f}")
    print(classification_report(y, pred, labels=list(range(len(classes))), target_names=classes, zero_division=0, digits=3))
    print("confusion matrix (hàng = thực tế, cột = dự đoán):")
    width = max(len(c) for c in classes) + 2
    print(" " * width + "".join(f"{c:>{width}}" for c in classes))
    for c, row in zip(classes, cm):
        print(f"{c:>{width}}" + "".join(f"{v:>{width}}" for v in row))
    plot_cm(cm, classes, f"{name} — acc {acc:.2f}, F1 {f1m:.2f}", out_dir / f"baseline_cm_{name}.png")
    return {"model": name, "accuracy": acc, "macro_f1": f1m}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", type=Path, default=Path("data/processed/features.npz"))
    p.add_argument("--classes", default="im,m,om")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("models"))
    a = p.parse_args(argv)
    classes = [c.strip() for c in a.classes.split(",") if c.strip()]
    a.out.mkdir(parents=True, exist_ok=True)

    X, y, groups = load_features(a.features, classes)
    fruits_per_class = Counter(classes[int(c)] for c in {g: c for g, c in zip(groups.tolist(), y.tolist())}.values())
    print(f"Dữ liệu: {len(y)} mẫu, {len(set(groups.tolist()))} quả, {X.shape[1]} đặc trưng")
    print("Quả / lớp:", dict(fruits_per_class))
    for c, n in fruits_per_class.items():
        if n < MIN_FRUITS_WARN:
            print(f"  [cảnh báo] lớp '{c}' chỉ có {n} quả -> F1 lớp này sẽ rất nhiễu, cần thu thêm dữ liệu.")

    summary = []
    for name, model in make_models(a.seed).items():
        pred, fold_f1 = cross_validate(model, X, y, groups, a.folds, a.seed)
        summary.append(report(name, y, pred, classes, fold_f1, a.out))
        model.fit(X, y)  # fit lại trên toàn bộ dữ liệu để lưu
        joblib.dump({"model": model, "classes": classes}, a.out / f"baseline_{name}.joblib")

    print("\n===== TÓM TẮT =====")
    print(f"{'model':<6}{'accuracy':>10}{'macro-F1':>10}")
    for s in summary:
        print(f"{s['model']:<6}{s['accuracy']:>10.3f}{s['macro_f1']:>10.3f}")
    print(f"Đã lưu model và confusion matrix PNG vào {a.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

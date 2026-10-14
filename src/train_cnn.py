"""CNN nhỏ trên log-mel spectrogram, export TFLite int8 cho ESP32-S3 / Pi Zero 2 W.

Kiến trúc (~24k tham số, < 100 KB sau quantize):
  Input (40 mel, 51 frame, 1)
  -> [Conv2D 3x3 -> BN -> ReLU -> MaxPool 2x2] x3 (16, 32, 64 kênh)
  -> GlobalAveragePooling -> Dropout -> Dense(n_classes, softmax)

Chuẩn hoá đầu vào: log-mel (dB, ref=max, top_db=80) nằm trong [-80, 0] -> (x + 40) / 40 ∈ [-1, 1].
Front-end trên thiết bị phải tái tạo đúng bước này.

Chia dữ liệu theo quả (StratifiedGroupKFold): fold --test-fold làm test, 1 fold trong phần còn lại làm val
(early stopping). --cv chạy thêm K-fold float-only để có ước lượng ổn định hơn.

Augmentation (chỉ train): dịch thời gian ±1 frame, nhiễu Gaussian trên dB, SpecAugment nhẹ (1 dải tần, 1 dải thời gian).

Chạy:
  python -m src.train_cnn --features data/processed/features.npz --classes im,m,om --epochs 150
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf

from src.features import N_FRAMES, N_MELS
from src.train_baseline import load_features as _load_mfcc  # dùng lại logic lọc lớp

MEL_OFFSET, MEL_SCALE = 40.0, 40.0


# --------------------------------------------------------------------------- data
def load_mel_features(path: Path, classes: list[str]):
    d = np.load(path, allow_pickle=False)
    file_classes = [str(c) for c in d["classes"]]
    missing = [c for c in classes if c not in file_classes]
    if missing:
        raise SystemExit(f"Lớp {missing} không có trong {path} (có: {file_classes}).")
    remap = {file_classes.index(c): i for i, c in enumerate(classes)}
    keep = np.isin(d["y"], list(remap))
    y = np.array([remap[int(v)] for v in d["y"][keep]], dtype=np.int64)
    X = d["X_mel"][keep].astype(np.float32)
    return X, y, d["groups"][keep]


def normalize_mel(X: np.ndarray) -> np.ndarray:
    return ((X + MEL_OFFSET) / MEL_SCALE).astype(np.float32)[..., None]  # (N, 40, 51, 1)


def pick_test_fold(y, groups, n_folds: int, seed: int, n_classes: int) -> int:
    """Fold test có số mẫu của lớp ít nhất là lớn nhất (tránh fold không có lớp `im`)."""
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    best, best_min = 0, -1
    for k, (_, te) in enumerate(cv.split(np.zeros(len(y)), y, groups)):
        counts = np.bincount(y[te], minlength=n_classes)
        if counts.min() > best_min:
            best, best_min = k, int(counts.min())
    return best


def split_train_val_test(y, groups, test_fold: int, n_folds: int, seed: int):
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = list(cv.split(np.zeros(len(y)), y, groups))
    trval, te = folds[test_fold % n_folds]
    inner = StratifiedGroupKFold(n_splits=max(2, n_folds - 1), shuffle=True, random_state=seed)
    tr_i, va_i = next(inner.split(np.zeros(len(trval)), y[trval], groups[trval]))
    return trval[tr_i], trval[va_i], te


def augment(x: tf.Tensor, y: tf.Tensor):
    """x: (40, 51, 1) đã chuẩn hoá. Áp dụng ngẫu nhiên trong tf.data."""
    shift = tf.random.uniform([], -1, 2, dtype=tf.int32)  # -1, 0, +1 frame (10 ms)
    x = tf.roll(x, shift, axis=1)
    x = x + tf.random.normal(tf.shape(x), stddev=0.03)  # ~1.2 dB
    # SpecAugment nhẹ: 1 dải tần tối đa 4 mel, 1 dải thời gian tối đa 5 frame
    f0 = tf.random.uniform([], 0, N_MELS - 4, dtype=tf.int32)
    fw = tf.random.uniform([], 0, 5, dtype=tf.int32)
    t0 = tf.random.uniform([], 0, N_FRAMES - 5, dtype=tf.int32)
    tw = tf.random.uniform([], 0, 6, dtype=tf.int32)
    fi = tf.range(N_MELS)[:, None, None]
    ti = tf.range(N_FRAMES)[None, :, None]
    mask = tf.logical_or(tf.logical_and(fi >= f0, fi < f0 + fw), tf.logical_and(ti >= t0, ti < t0 + tw))
    x = tf.where(mask, tf.zeros_like(x), x)
    return x, y


def make_ds(X, y, batch: int, train: bool, seed: int) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if train:
        ds = ds.shuffle(len(y), seed=seed, reshuffle_each_iteration=True).map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


# -------------------------------------------------------------------------- model
def build_model(n_classes: int, width: tuple[int, int, int] = (16, 32, 64), dropout: float = 0.3) -> tf.keras.Model:
    L = tf.keras.layers
    inp = L.Input(shape=(N_MELS, N_FRAMES, 1), name="log_mel")
    x = inp
    for i, c in enumerate(width):
        x = L.Conv2D(c, 3, padding="same", use_bias=False, name=f"conv{i + 1}")(x)
        x = L.BatchNormalization(name=f"bn{i + 1}")(x)
        x = L.ReLU(name=f"relu{i + 1}")(x)
        x = L.MaxPool2D(2, name=f"pool{i + 1}")(x)
    x = L.GlobalAveragePooling2D(name="gap")(x)
    x = L.Dropout(dropout, name="drop")(x)
    out = L.Dense(n_classes, activation="softmax", name="probs")(x)
    return tf.keras.Model(inp, out, name="coconut_tap_cnn")


def train_one(X, y, groups, tr, va, classes, a, verbose=0) -> tf.keras.Model:
    tf.keras.utils.set_random_seed(a.seed)
    model = build_model(len(classes), dropout=a.dropout)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(a.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    cw = compute_class_weight("balanced", classes=np.arange(len(classes)), y=y[tr])
    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=a.patience, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(5, a.patience // 3), min_lr=1e-5),
    ]
    model.fit(
        make_ds(X[tr], y[tr], a.batch, True, a.seed),
        validation_data=make_ds(X[va], y[va], a.batch, False, a.seed),
        epochs=a.epochs,
        class_weight={i: float(w) for i, w in enumerate(cw)},
        callbacks=cbs,
        verbose=verbose,
    )
    return model


# ------------------------------------------------------------------------- tflite
def export_tflite_int8(model: tf.keras.Model, X_rep: np.ndarray, out: Path) -> bytes:
    def rep():
        for i in range(0, len(X_rep), 1):
            yield [X_rep[i : i + 1].astype(np.float32)]

    # Keras 3 + TF 2.16+: export SavedModel rồi convert (from_keras_model in ra nhiều cảnh báo resource)
    import shutil
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="coco_sm_"))
    try:
        model.export(str(tmp), verbose=False)
        conv = tf.lite.TFLiteConverter.from_saved_model(str(tmp))
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = rep
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
        blob = conv.convert()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    out.write_bytes(blob)
    return blob


def tflite_predict(blob: bytes, X: np.ndarray) -> np.ndarray:
    it = tf.lite.Interpreter(model_content=blob)
    it.allocate_tensors()
    inp, outp = it.get_input_details()[0], it.get_output_details()[0]
    s_in, z_in = inp["quantization"]
    preds = []
    for i in range(len(X)):
        xq = np.clip(np.round(X[i : i + 1] / s_in + z_in), -128, 127).astype(np.int8)
        it.set_tensor(inp["index"], xq)
        it.invoke()
        preds.append(int(np.argmax(it.get_tensor(outp["index"])[0])))
    return np.asarray(preds)


def tflite_arena_estimate(blob: bytes) -> int:
    """Ước lượng thô tensor arena cho TFLM: tổng kích thước 2 tensor trung gian lớn nhất + 10%."""
    it = tf.lite.Interpreter(model_content=blob)
    it.allocate_tensors()
    sizes = sorted((int(np.prod(t["shape"])) * np.dtype(t["dtype"]).itemsize for t in it.get_tensor_details() if len(t["shape"])), reverse=True)
    return int(sum(sizes[:2]) * 1.1)


def write_c_array(blob: bytes, out: Path, name: str = "g_coconut_model") -> None:
    lines = [f"// Sinh tự động từ {out.with_suffix('.tflite').name}; dùng với TFLite Micro.",
             "#include <cstdint>", "", f"alignas(16) const unsigned char {name}[] = {{"]
    for i in range(0, len(blob), 12):
        lines.append("  " + ", ".join(f"0x{b:02x}" for b in blob[i : i + 12]) + ",")
    lines += ["};", f"const unsigned int {name}_len = {len(blob)};", ""]
    out.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- main
def metrics(y, p, n_classes):
    return accuracy_score(y, p), f1_score(y, p, average="macro", zero_division=0), confusion_matrix(y, p, labels=list(range(n_classes)))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", type=Path, default=Path("data/processed/features.npz"))
    p.add_argument("--classes", default="im,m,om")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--test-fold", default="auto", help="chỉ số fold làm test, hoặc 'auto' = fold có đủ mọi lớp nhiều nhất")
    p.add_argument("--cv", action="store_true", help="chạy thêm K-fold float-only để ước lượng ổn định")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("models"))
    p.add_argument("--name", default="coconut_cnn")
    a = p.parse_args(argv)
    classes = [c.strip() for c in a.classes.split(",") if c.strip()]
    a.out.mkdir(parents=True, exist_ok=True)

    X_raw, y, groups = load_mel_features(a.features, classes)
    X = normalize_mel(X_raw)
    n = len(classes)
    print(f"Dữ liệu: {X.shape} -> {n} lớp {classes}, {len(set(groups.tolist()))} quả")
    print(f"Tham số model: {build_model(n).count_params():,}")

    if a.cv:
        print(f"\n===== CV {a.folds}-fold (float) =====")
        oof = np.full_like(y, -1)
        for k in range(a.folds):
            tr, va, te = split_train_val_test(y, groups, k, a.folds, a.seed)
            m = train_one(X, y, groups, tr, va, classes, a)
            oof[te] = np.argmax(m.predict(X[te], verbose=0), axis=1)
            acc_k, f1_k, _ = metrics(y[te], oof[te], n)
            print(f"fold {k}: acc {acc_k:.3f}  macro-F1 {f1_k:.3f}  (test {len(te)} mẫu)")
        acc, f1m, cm = metrics(y, oof, n)
        print(f"OOF: acc {acc:.3f}  macro-F1 {f1m:.3f}\n{cm}")

    test_fold = pick_test_fold(y, groups, a.folds, a.seed, n) if a.test_fold == "auto" else int(a.test_fold)
    print(f"\n===== Train cuối (test = fold {test_fold}) =====")
    tr, va, te = split_train_val_test(y, groups, test_fold, a.folds, a.seed)
    print(f"train {len(tr)} / val {len(va)} / test {len(te)} mẫu; test theo lớp: {dict(zip(classes, np.bincount(y[te], minlength=n).tolist()))}")
    model = train_one(X, y, groups, tr, va, classes, a, verbose=0)
    keras_path = a.out / f"{a.name}.keras"
    model.save(keras_path)

    p_float = np.argmax(model.predict(X[te], verbose=0), axis=1)
    acc_f, f1_f, cm_f = metrics(y[te], p_float, n)

    tflite_path = a.out / f"{a.name}_int8.tflite"
    blob = export_tflite_int8(model, X[tr], tflite_path)
    p_q = tflite_predict(blob, X[te])
    acc_q, f1_q, cm_q = metrics(y[te], p_q, n)
    write_c_array(blob, a.out / f"{a.name}_int8.cc")

    print("\n===== KẾT QUẢ (test set, chia theo quả) =====")
    print(f"{'':<14}{'accuracy':>10}{'macro-F1':>10}")
    print(f"{'float32':<14}{acc_f:>10.3f}{f1_f:>10.3f}")
    print(f"{'int8 tflite':<14}{acc_q:>10.3f}{f1_q:>10.3f}")
    print(f"chênh lệch acc : {acc_q - acc_f:+.3f}")
    print("\nconfusion float32:\n", cm_f, "\nconfusion int8:\n", cm_q)
    print(f"\nkích thước .tflite : {len(blob):,} bytes ({len(blob) / 1024:.1f} KB)")
    print(f"tham số            : {model.count_params():,}")
    print(f"tensor arena (ước) : ~{tflite_arena_estimate(blob) / 1024:.0f} KB")
    print(f"đã lưu: {keras_path}, {tflite_path}, {tflite_path.with_suffix('.cc')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

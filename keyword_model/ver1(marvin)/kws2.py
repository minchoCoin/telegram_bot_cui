#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


# =========================================================
# 1. settings
# =========================================================
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

ROOT_DIR = Path(".")

NORMAL_CLASS = "marvin"
ABNORMAL_CLASSES = ["bed", "bird", "nine", "left"]
TARGET_CLASSES = [NORMAL_CLASS] + ABNORMAL_CLASSES

SAMPLE_RATE = 16000
AUDIO_LENGTH = 16000
N_MELS = 64
N_FFT = 512
HOP_LENGTH = 160
WIN_LENGTH = 400
FMIN = 20
FMAX = SAMPLE_RATE // 2

BATCH_SIZE = 1
EPOCHS = 10
LEARNING_RATE = 1e-3
BACKGROUND_CLASS = "_background_noise_"
BACKGROUND_SPLIT_RATIOS = (0.8, 0.1, 0.1)


# =========================================================
# 2. split files
# =========================================================
def read_split_file(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        return set(line.strip().replace("\\", "/") for line in f if line.strip())


val_set = read_split_file(ROOT_DIR / "validation_list.txt")
test_set = read_split_file(ROOT_DIR / "testing_list.txt")

print(f"#validation files: {len(val_set)}")
print(f"#testing files   : {len(test_set)}")


# =========================================================
# 3. collect wav files
# =========================================================
def collect_files_by_split(root_dir):
    train_files, val_files, test_files = [], [], []

    for cls in TARGET_CLASSES:
        cls_dir = root_dir / cls
        if not cls_dir.exists():
            print(f"[WARN] class folder not found: {cls_dir}")
            continue

        for wav_path in cls_dir.glob("*.wav"):
            rel_path = wav_path.relative_to(root_dir).as_posix()
            item = (wav_path, cls, 0.0)

            if rel_path in val_set:
                val_files.append(item)
            elif rel_path in test_set:
                test_files.append(item)
            else:
                train_files.append(item)

    return train_files, val_files, test_files


def collect_background_segments(root_dir):
    train_files, val_files, test_files = [], [], []
    bg_dir = root_dir / BACKGROUND_CLASS
    if not bg_dir.exists():
        print(f"[WARN] background folder not found: {bg_dir}")
        return train_files, val_files, test_files

    train_ratio, val_ratio, _ = BACKGROUND_SPLIT_RATIOS

    for wav_path in bg_dir.glob("*.wav"):
        audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
        total_segments = len(audio) // AUDIO_LENGTH

        for seg_idx in range(total_segments):
            offset_sec = seg_idx * AUDIO_LENGTH / SAMPLE_RATE
            item = (wav_path, BACKGROUND_CLASS, offset_sec)
            progress = seg_idx / max(total_segments, 1)

            if progress < train_ratio:
                train_files.append(item)
            elif progress < train_ratio + val_ratio:
                val_files.append(item)
            else:
                test_files.append(item)

    return train_files, val_files, test_files


train_files, val_files, test_files = collect_files_by_split(ROOT_DIR)
bg_train_files, bg_val_files, bg_test_files = collect_background_segments(ROOT_DIR)

train_files.extend(bg_train_files)
val_files.extend(bg_val_files)
test_files.extend(bg_test_files)

print(f"Collected train: {len(train_files)}")
print(f"Collected val  : {len(val_files)}")
print(f"Collected test : {len(test_files)}")


# =========================================================
# 4. log-mel feature extraction
# =========================================================
def pad_or_trim(audio, target_len=AUDIO_LENGTH):
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        audio = audio[:target_len]
    return audio


def extract_log_mel_librosa(wav_path, offset_sec=0.0):
    audio, _ = librosa.load(
        wav_path,
        sr=SAMPLE_RATE,
        mono=True,
        offset=offset_sec,
        duration=AUDIO_LENGTH / SAMPLE_RATE,
    )
    audio = pad_or_trim(audio, AUDIO_LENGTH)

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
        center=False,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = log_mel.T

    mean = log_mel.mean()
    std = log_mel.std() + 1e-6
    log_mel = (log_mel - mean) / std

    return log_mel.astype(np.float32)


# =========================================================
# 5. build arrays
# =========================================================
def binary_label(cls_name):
    return 1 if cls_name == NORMAL_CLASS else 0


def build_dataset(file_list):
    X, y, names = [], [], []

    for wav_path, cls, offset_sec in file_list:
        X.append(extract_log_mel_librosa(wav_path, offset_sec))
        y.append(binary_label(cls))
        names.append(cls)

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.int32),
        np.asarray(names),
    )


print("Extracting train features...")
X_train, y_train, names_train = build_dataset(train_files)

print("Extracting val features...")
X_val, y_val, names_val = build_dataset(val_files)

print("Extracting test features...")
X_test, y_test, names_test = build_dataset(test_files)

print("X_train:", X_train.shape, y_train.shape)
print("X_val  :", X_val.shape, y_val.shape)
print("X_test :", X_test.shape, y_test.shape)


# =========================================================
# 6. tf.data
# =========================================================
train_ds = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .shuffle(len(X_train), seed=SEED)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((X_val, y_val))
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

test_ds = (
    tf.data.Dataset.from_tensor_slices((X_test, y_test))
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)


# =========================================================
# 7. LSTM classifier
# =========================================================
def build_lstm_classifier(input_shape):
    inputs = tf.keras.Input(shape=input_shape, batch_size=BATCH_SIZE, name="logmel_input")

    #x = tf.keras.layers.Masking(mask_value=0.0)(inputs)
    x = tf.keras.layers.LSTM(128, return_sequences=True)(inputs)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.LSTM(64)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax")(x)

    return tf.keras.Model(inputs, outputs, name="lstm_logmel_classifier")


input_shape = X_train.shape[1:]
model = build_lstm_classifier(input_shape)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()


# =========================================================
# 8. train
# =========================================================
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1,
)


# =========================================================
# 8-1. export tflite(float32)
# =========================================================
def export_tflite_float32(model, output_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        #tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    #converter._experimental_lower_tensor_list_ops = False

    tflite_model = converter.convert()
    output_path.write_bytes(tflite_model)
    print(f"Saved TFLite model: {output_path}")


tflite_output_path = ROOT_DIR / "kws2_float32.tflite"
export_tflite_float32(model, tflite_output_path)


# =========================================================
# 9. prediction helpers
# =========================================================
def collect_predictions(model, dataset):
    positive_probs, pred_labels, true_labels = [], [], []

    for xb, yb in dataset:
        pred = model.predict(xb, verbose=0)
        positive_probs.extend(pred[:, 1].tolist())
        pred_labels.extend(np.argmax(pred, axis=1).tolist())
        true_labels.extend(yb.numpy().tolist())

    return (
        np.asarray(positive_probs),
        np.asarray(pred_labels, dtype=np.int32),
        np.asarray(true_labels, dtype=np.int32),
    )


# =========================================================
# 10. evaluation
# =========================================================
def evaluate(model, dataset, split_name="eval"):
    probs, preds, labels = collect_predictions(model, dataset)

    print(f"\n[{split_name}]")
    print("Confusion Matrix:")
    print(confusion_matrix(labels, preds))
    print("\nClassification Report:")
    print(classification_report(labels, preds, digits=4, zero_division=0))

    try:
        auc = roc_auc_score(labels, probs)
        print(f"ROC-AUC: {auc:.4f}")
    except Exception as e:
        print("ROC-AUC failed:", e)

    return probs, labels, preds


val_probs, val_labels, val_preds = evaluate(model, val_ds, "validation")
test_probs, test_labels, test_preds = evaluate(model, test_ds, "test")


# =========================================================
# 11. class-wise positive probability
# =========================================================
def print_classwise_probability(probs, class_names):
    print(f"\nClass-wise mean P({NORMAL_CLASS}):")
    unique_classes = sorted(set(class_names.tolist()))
    for cls in unique_classes:
        mask = class_names == cls
        mean_prob = probs[mask].mean()
        print(f"{cls:>8s}: {mean_prob:.6f}")


print_classwise_probability(val_probs, names_val)
print_classwise_probability(test_probs, names_test)


# =========================================================
# 12. plot test probability distribution
# =========================================================
def plot_test_probability_distribution(probs, labels, class_names):
    normal_probs = probs[labels == 1]
    abnormal_probs = probs[labels == 0]

    plt.figure(figsize=(12, 6))
    plt.hist(normal_probs, bins=50, alpha=0.6, label=f"{NORMAL_CLASS} prob")
    plt.hist(abnormal_probs, bins=50, alpha=0.6, label="not_marvin prob")
    plt.title("Test Positive-Class Probability Distribution")
    plt.xlabel(f"P({NORMAL_CLASS})")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.show()

    unique_classes = sorted(set(class_names.tolist()))
    class_means = [probs[class_names == cls].mean() for cls in unique_classes]

    plt.figure(figsize=(12, 6))
    plt.bar(unique_classes, class_means)
    plt.title(f"Mean P({NORMAL_CLASS}) by Test Class")
    plt.xlabel("Class")
    plt.ylabel("Mean Positive Probability")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()


plot_test_probability_distribution(test_probs, test_labels, names_test)

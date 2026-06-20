"""
Member 5 - Deep Learning and Tuning.

This module trains a text classifier from the labeled data created by member 4.
It is intentionally self-contained so importing the main pipeline does not
trigger training or require TensorFlow unless the training step is executed.
"""

import json
import os

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder


DEFAULT_TRAINING_CONFIG = {
    "train_path": "data/train.csv",
    "val_path": "data/val.csv",
    "test_path": "data/test.csv",
    "fallback_labeled_path": "data/labeled_news.csv",
    "fallback_clustered_path": "data/clustered_news.csv",
    "vectorizer_path": "models/vectorizer.pkl",
    "model_path": "models/deep_learning_text_model.pkl",
    "metrics_path": "models/deep_learning_metrics.json",
    "confusion_matrix_path": "data/deep_learning_confusion_matrix.csv",
    "confusion_matrix_plot_path": "notebooks/deep_learning_01_confusion_matrix.png",
    "metrics_plot_path": "notebooks/deep_learning_02_metrics.png",
    "class_f1_plot_path": "notebooks/deep_learning_03_class_f1.png",
    "dataset_split_plot_path": "notebooks/deep_learning_04_dataset_split.png",
    "text_column": "processed_content",
    "fallback_text_column": "main_content",
    "label_column": "label",
    "backend": "sklearn_mlp",
    "hidden_layer_sizes": [128, 64],
    "learning_rate_init": 0.001,
    "batch_size": 64,
    "max_iter": 30,
    "random_state": 42,
}


def _project_path(project_root, relative_path):
    return os.path.join(project_root, relative_path)


def _merge_config(config):
    merged = DEFAULT_TRAINING_CONFIG.copy()
    if config:
        merged.update(config)
    return merged


def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _read_csv_if_exists(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def _pick_text_column(df, config):
    preferred = config.get("text_column", "processed_content")
    fallback = config.get("fallback_text_column", "main_content")
    if preferred in df.columns:
        return preferred
    if fallback in df.columns:
        return fallback
    raise ValueError(
        "Khong tim thay cot van ban. Can co cot "
        f"'{preferred}' hoac '{fallback}'."
    )


def _load_training_frames(project_root, config):
    train_path = _project_path(project_root, config["train_path"])
    val_path = _project_path(project_root, config["val_path"])
    test_path = _project_path(project_root, config["test_path"])

    train_df = _read_csv_if_exists(train_path)
    val_df = _read_csv_if_exists(val_path)
    test_df = _read_csv_if_exists(test_path)

    if train_df is not None and val_df is not None and test_df is not None:
        return train_df, val_df, test_df

    labeled_paths = [
        _project_path(project_root, config["fallback_labeled_path"]),
        _project_path(project_root, config["fallback_clustered_path"]),
    ]
    labeled_df = None
    for path in labeled_paths:
        labeled_df = _read_csv_if_exists(path)
        if labeled_df is not None:
            break

    if labeled_df is None:
        return None, None, None

    label_column = config["label_column"]
    if label_column not in labeled_df.columns:
        return None, None, None

    from sklearn.model_selection import train_test_split

    train_df, temp_df = train_test_split(
        labeled_df,
        test_size=0.30,
        stratify=labeled_df[label_column],
        random_state=config["random_state"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df[label_column],
        random_state=config["random_state"],
    )
    return train_df, val_df, test_df


def _load_vectorizer(project_root, config):
    vectorizer_path = _project_path(project_root, config["vectorizer_path"])
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(
            f"Khong tim thay vectorizer: {vectorizer_path}. "
            "Hay chay buoc tien xu ly NLP truoc."
        )
    return joblib.load(vectorizer_path)


def _train_sklearn_mlp(X_train, y_train, config):
    model = MLPClassifier(
        hidden_layer_sizes=tuple(config["hidden_layer_sizes"]),
        learning_rate_init=config["learning_rate_init"],
        batch_size=config["batch_size"],
        max_iter=config["max_iter"],
        random_state=config["random_state"],
        early_stopping=True,
        validation_fraction=0.1,
    )
    model.fit(X_train, y_train)
    return model


def _train_tensorflow_dense(X_train, y_train, X_val, y_val, config):
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow chua duoc cai. Cai tensorflow hoac doi backend "
            "sang 'sklearn_mlp'."
        ) from exc

    X_train_dense = X_train.toarray() if hasattr(X_train, "toarray") else X_train
    X_val_dense = X_val.toarray() if hasattr(X_val, "toarray") else X_val

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(X_train_dense.shape[1],)),
            tf.keras.layers.Dense(config["hidden_layer_sizes"][0], activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(config["hidden_layer_sizes"][-1], activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(len(set(y_train)), activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=config["learning_rate_init"]
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        X_train_dense,
        y_train,
        validation_data=(X_val_dense, y_val),
        epochs=config.get("epochs", 8),
        batch_size=config["batch_size"],
        verbose=1,
    )
    return model


def _predict(model, X, backend):
    if backend == "tensorflow_dense":
        X_dense = X.toarray() if hasattr(X, "toarray") else X
        return model.predict(X_dense, verbose=0).argmax(axis=1)
    return model.predict(X)


def save_confusion_matrix_plot(cm, labels, output_path):
    """Save a confusion matrix heatmap for the TV5 report."""
    import matplotlib.pyplot as plt

    _ensure_parent_dir(output_path)
    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title("Deep Learning Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_metrics_plot(metrics, output_path):
    """Save Accuracy/F1 chart for the TV5 report."""
    import matplotlib.pyplot as plt

    _ensure_parent_dir(output_path)
    names = ["Val Accuracy", "Val F1", "Test Accuracy", "Test F1"]
    values = [
        metrics.get("val_accuracy", 0),
        metrics.get("val_f1_macro", 0),
        metrics.get("test_accuracy", 0),
        metrics.get("test_f1_macro", 0),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, values, color=["#2F80ED", "#56CCF2", "#27AE60", "#6FCF97"])
    ax.set_title("Deep Learning Evaluation Metrics")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(value + 0.02, 0.98),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_class_f1_plot(metrics, output_path):
    """Save per-class F1-score chart for the deep learning report."""
    import matplotlib.pyplot as plt

    report = metrics.get("classification_report", {})
    class_names = [
        name
        for name, values in report.items()
        if isinstance(values, dict) and "f1-score" in values
    ]
    class_names = [
        name
        for name in class_names
        if name not in {"accuracy", "macro avg", "weighted avg"}
    ]
    f1_scores = [report[name]["f1-score"] for name in class_names]

    _ensure_parent_dir(output_path)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(class_names, f1_scores, color="#9B51E0")
    ax.set_title("Deep Learning F1-score by Class")
    ax.set_xlabel("F1-score")
    ax.set_xlim(0, 1)
    ax.grid(axis="x", alpha=0.25)

    for bar, value in zip(bars, f1_scores):
        ax.text(
            min(value + 0.02, 0.98),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_dataset_split_plot(metrics, output_path):
    """Save train/validation/test size chart for the deep learning report."""
    import matplotlib.pyplot as plt

    names = ["Train", "Validation", "Test"]
    values = [
        metrics.get("train_size", 0),
        metrics.get("val_size", 0),
        metrics.get("test_size", 0),
    ]

    _ensure_parent_dir(output_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, values, color=["#2D9CDB", "#F2C94C", "#EB5757"])
    ax.set_title("Deep Learning Dataset Split")
    ax.set_ylabel("Number of Articles")
    ax.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:,}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_existing_results(project_root=None, config=None):
    """Create TV5 PNG charts from existing metrics JSON and confusion matrix CSV."""
    project_root = project_root or os.path.dirname(os.path.dirname(__file__))
    config = _merge_config(config)

    metrics_path = _project_path(project_root, config["metrics_path"])
    cm_path = _project_path(project_root, config["confusion_matrix_path"])
    cm_plot_path = _project_path(project_root, config["confusion_matrix_plot_path"])
    metrics_plot_path = _project_path(project_root, config["metrics_plot_path"])
    class_f1_plot_path = _project_path(project_root, config["class_f1_plot_path"])
    dataset_split_plot_path = _project_path(project_root, config["dataset_split_plot_path"])

    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Khong tim thay metrics: {metrics_path}")
    if not os.path.exists(cm_path):
        raise FileNotFoundError(f"Khong tim thay confusion matrix: {cm_path}")

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    cm_df = pd.read_csv(cm_path, index_col=0)
    labels = list(cm_df.index.astype(str))
    cm = cm_df.to_numpy()

    save_confusion_matrix_plot(cm, labels, cm_plot_path)
    save_metrics_plot(metrics, metrics_plot_path)
    save_class_f1_plot(metrics, class_f1_plot_path)
    save_dataset_split_plot(metrics, dataset_split_plot_path)

    return {
        "confusion_matrix_plot_path": cm_plot_path,
        "metrics_plot_path": metrics_plot_path,
        "class_f1_plot_path": class_f1_plot_path,
        "dataset_split_plot_path": dataset_split_plot_path,
    }


def train_deep_learning_model(project_root=None, config=None):
    """
    Train member 5's classifier.

    Returns:
        dict: status and metrics. If member 4 data is not available yet,
        status is "skipped" instead of raising an error.
    """
    project_root = project_root or os.path.dirname(os.path.dirname(__file__))
    config = _merge_config(config)

    train_df, val_df, test_df = _load_training_frames(project_root, config)
    if train_df is None or val_df is None or test_df is None:
        return {
            "status": "skipped",
            "reason": "Chua co du lieu gan nhan tu thanh vien 4.",
        }

    label_column = config["label_column"]
    if label_column not in train_df.columns:
        return {
            "status": "skipped",
            "reason": f"Khong tim thay cot nhan '{label_column}'.",
        }

    text_column = _pick_text_column(train_df, config)
    vectorizer = _load_vectorizer(project_root, config)

    X_train = vectorizer.transform(train_df[text_column].fillna(""))
    X_val = vectorizer.transform(val_df[text_column].fillna(""))
    X_test = vectorizer.transform(test_df[text_column].fillna(""))

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train_df[label_column].astype(str))
    y_val = encoder.transform(val_df[label_column].astype(str))
    y_test = encoder.transform(test_df[label_column].astype(str))

    backend = config.get("backend", "sklearn_mlp")
    if backend == "tensorflow_dense":
        model = _train_tensorflow_dense(X_train, y_train, X_val, y_val, config)
    else:
        backend = "sklearn_mlp"
        model = _train_sklearn_mlp(X_train, y_train, config)

    y_val_pred = _predict(model, X_val, backend)
    y_test_pred = _predict(model, X_test, backend)

    metrics = {
        "status": "completed",
        "backend": backend,
        "text_column": text_column,
        "label_column": label_column,
        "train_size": int(len(train_df)),
        "val_size": int(len(val_df)),
        "test_size": int(len(test_df)),
        "val_accuracy": float(accuracy_score(y_val, y_val_pred)),
        "val_f1_macro": float(f1_score(y_val, y_val_pred, average="macro")),
        "test_accuracy": float(accuracy_score(y_test, y_test_pred)),
        "test_f1_macro": float(f1_score(y_test, y_test_pred, average="macro")),
        "classification_report": classification_report(
            y_test,
            y_test_pred,
            target_names=encoder.classes_,
            output_dict=True,
            zero_division=0,
        ),
    }

    model_path = _project_path(project_root, config["model_path"])
    metrics_path = _project_path(project_root, config["metrics_path"])
    cm_path = _project_path(project_root, config["confusion_matrix_path"])
    cm_plot_path = _project_path(project_root, config["confusion_matrix_plot_path"])
    metrics_plot_path = _project_path(project_root, config["metrics_plot_path"])
    class_f1_plot_path = _project_path(project_root, config["class_f1_plot_path"])
    dataset_split_plot_path = _project_path(project_root, config["dataset_split_plot_path"])

    _ensure_parent_dir(model_path)
    _ensure_parent_dir(metrics_path)
    _ensure_parent_dir(cm_path)

    if backend == "tensorflow_dense":
        keras_path = model_path.replace(".pkl", ".keras")
        model.save(keras_path)
        model_payload = {
            "backend": backend,
            "keras_model_path": os.path.relpath(keras_path, project_root),
            "label_encoder": encoder,
            "text_column": text_column,
            "label_column": label_column,
        }
    else:
        model_payload = {
            "backend": backend,
            "model": model,
            "label_encoder": encoder,
            "text_column": text_column,
            "label_column": label_column,
        }

    joblib.dump(model_payload, model_path)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    cm = confusion_matrix(y_test, y_test_pred)
    pd.DataFrame(cm, index=encoder.classes_, columns=encoder.classes_).to_csv(
        cm_path,
        encoding="utf-8-sig",
    )
    save_confusion_matrix_plot(cm, encoder.classes_, cm_plot_path)
    save_metrics_plot(metrics, metrics_plot_path)
    save_class_f1_plot(metrics, class_f1_plot_path)
    save_dataset_split_plot(metrics, dataset_split_plot_path)

    return metrics


if __name__ == "__main__":
    result = train_deep_learning_model()
    print(json.dumps(result, ensure_ascii=False, indent=2))

"""Train, self-train, evaluate, and save the news topic classifier.

Member 5 scope:
- Calibrate the base classifier with CalibratedClassifierCV.
- Generate pseudo labels for unlabeled samples with high confidence.
- Retrain on labeled data + pseudo-labeled data.
- Evaluate with Accuracy, F1-score, and Confusion Matrix.
- Save final_model_calibrated.pkl and pseudo_labels.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

DEFAULT_TRAIN_PATH = DATA_DIR / "train_data.json"
DEFAULT_UNLABELED_PATH = DATA_DIR / "unlabeled_data.json"
DEFAULT_MODEL_PATH = MODELS_DIR / "final_model_calibrated.pkl"
DEFAULT_PSEUDO_LABEL_PATH = MODELS_DIR / "pseudo_labels.json"
DEFAULT_REPORT_PATH = MODELS_DIR / "evaluation_report.json"

TEXT_COLUMN_CANDIDATES = (
    "text",
    "content",
    "article_content",
    "clean_text",
    "description",
    "title",
)
LABEL_COLUMN_CANDIDATES = ("label", "category", "topic", "subject")


def load_json_records(path: Path) -> pd.DataFrame:
    """Load either JSON list records or JSONL into a DataFrame."""
    if not path.exists():
        raise FileNotFoundError(f"Khong tim thay file du lieu: {path}")

    try:
        return pd.read_json(path)
    except ValueError:
        return pd.read_json(path, lines=True)


def find_column(df: pd.DataFrame, candidates: tuple[str, ...], column_name: str) -> str:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        f"Khong tim thay cot {column_name}. Hay doi ten cot thanh mot trong: "
        f"{', '.join(candidates)}"
    )


def combine_text_columns(df: pd.DataFrame) -> pd.Series:
    """Use available text fields, joining title and description when useful."""
    available = [col for col in TEXT_COLUMN_CANDIDATES if col in df.columns]
    if not available:
        raise ValueError(
            "Khong tim thay cot van ban. Can co mot trong cac cot: "
            f"{', '.join(TEXT_COLUMN_CANDIDATES)}"
        )

    if "clean_text" in available:
        return df["clean_text"].fillna("").astype(str)

    preferred = [col for col in ("title", "description", "article_content", "content", "text") if col in available]
    return df[preferred].fillna("").astype(str).agg(" ".join, axis=1)


def build_classifier() -> Pipeline:
    """Build TF-IDF + SVD + calibrated Logistic Regression pipeline."""
    base_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )

    calibrated_model = CalibratedClassifierCV(
        estimator=base_model,
        method="sigmoid",
        cv=3,
    )

    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2))),
            ("svd", TruncatedSVD(n_components=300, random_state=42)),
            ("model", calibrated_model),
        ]
    )


def split_train_test(
    texts: pd.Series,
    labels: pd.Series,
    test_size: float = 0.2,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Create stratified train/test split for fair evaluation."""
    return train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=42,
        stratify=labels,
    )


def generate_pseudo_labels(
    model: Pipeline,
    unlabeled_texts: pd.Series,
    raw_unlabeled_df: pd.DataFrame,
    threshold: float = 0.9,
) -> pd.DataFrame:
    """Predict pseudo labels for high-confidence unlabeled articles."""
    probabilities = model.predict_proba(unlabeled_texts)
    predicted_labels = model.predict(unlabeled_texts)
    confidences = np.max(probabilities, axis=1)
    selected_mask = confidences >= threshold

    pseudo_df = raw_unlabeled_df.loc[selected_mask].copy()
    pseudo_df["pseudo_label"] = predicted_labels[selected_mask]
    pseudo_df["confidence"] = confidences[selected_mask]

    return pseudo_df


def evaluate_model(model: Pipeline, X_test: pd.Series, y_test: pd.Series) -> dict[str, Any]:
    """Evaluate final model with Accuracy, F1, confusion matrix, and report."""
    y_pred = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def save_pseudo_labels(pseudo_df: pd.DataFrame, path: Path) -> None:
    records = pseudo_df.to_dict(orient="records")
    save_json(records, path)


def run_self_training(
    train_path: Path = DEFAULT_TRAIN_PATH,
    unlabeled_path: Path = DEFAULT_UNLABELED_PATH,
    threshold: float = 0.9,
) -> dict[str, Any]:
    """Run the full member-5 training pipeline."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_json_records(train_path)
    unlabeled_df = load_json_records(unlabeled_path)

    label_column = find_column(train_df, LABEL_COLUMN_CANDIDATES, "nhan")
    labeled_texts = combine_text_columns(train_df)
    labels = train_df[label_column].fillna("").astype(str)
    unlabeled_texts = combine_text_columns(unlabeled_df)

    X_train, X_test, y_train, y_test = split_train_test(labeled_texts, labels)

    initial_model = build_classifier()
    initial_model.fit(X_train, y_train)

    pseudo_df = generate_pseudo_labels(
        initial_model,
        unlabeled_texts,
        unlabeled_df,
        threshold=threshold,
    )

    pseudo_texts = combine_text_columns(pseudo_df) if not pseudo_df.empty else pd.Series(dtype=str)
    pseudo_labels = pseudo_df["pseudo_label"] if not pseudo_df.empty else pd.Series(dtype=str)

    final_X_train = pd.concat([X_train.reset_index(drop=True), pseudo_texts.reset_index(drop=True)])
    final_y_train = pd.concat([y_train.reset_index(drop=True), pseudo_labels.reset_index(drop=True)])

    final_model = build_classifier()
    final_model.fit(final_X_train, final_y_train)

    metrics = evaluate_model(final_model, X_test, y_test)
    metrics["pseudo_label_count"] = int(len(pseudo_df))
    metrics["confidence_threshold"] = threshold

    joblib.dump(final_model, DEFAULT_MODEL_PATH)
    save_pseudo_labels(pseudo_df, DEFAULT_PSEUDO_LABEL_PATH)
    save_json(metrics, DEFAULT_REPORT_PATH)

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-training pipeline for Member 5.")
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--unlabeled-path", type=Path, default=DEFAULT_UNLABELED_PATH)
    parser.add_argument("--threshold", type=float, default=0.9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_self_training(
        train_path=args.train_path,
        unlabeled_path=args.unlabeled_path,
        threshold=args.threshold,
    )

    print("Hoan tat Self-Training.")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 weighted: {metrics['f1_weighted']:.4f}")
    print(f"So pseudo-label duoc chon: {metrics['pseudo_label_count']}")
    print(f"Da luu model: {DEFAULT_MODEL_PATH}")
    print(f"Da luu pseudo labels: {DEFAULT_PSEUDO_LABEL_PATH}")
    print(f"Da luu report: {DEFAULT_REPORT_PATH}")


if __name__ == "__main__":
    main()

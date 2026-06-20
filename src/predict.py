"""
Prediction utilities for member 5's trained text model.
"""

import os

import joblib


def _project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_model_bundle(project_root=None, model_path="models/deep_learning_text_model.pkl"):
    project_root = project_root or _project_root()
    full_path = os.path.join(project_root, model_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(
            f"Khong tim thay model: {full_path}. Hay chay buoc train_model truoc."
        )
    return joblib.load(full_path)


def _load_vectorizer(project_root=None, vectorizer_path="models/vectorizer.pkl"):
    project_root = project_root or _project_root()
    full_path = os.path.join(project_root, vectorizer_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(
            f"Khong tim thay vectorizer: {full_path}. Hay chay pipeline truoc."
        )
    return joblib.load(full_path)


def predict_texts(
    texts,
    project_root=None,
    model_path="models/deep_learning_text_model.pkl",
    vectorizer_path="models/vectorizer.pkl",
):
    """
    Predict labels for a list of processed texts.

    Text should be preprocessed in the same way as data/processed_news.csv.
    """
    if isinstance(texts, str):
        texts = [texts]

    project_root = project_root or _project_root()
    bundle = _load_model_bundle(project_root, model_path)
    vectorizer = _load_vectorizer(project_root, vectorizer_path)

    X = vectorizer.transform([text or "" for text in texts])
    backend = bundle.get("backend", "sklearn_mlp")

    if backend == "tensorflow_dense":
        import tensorflow as tf

        keras_path = os.path.join(project_root, bundle["keras_model_path"])
        model = tf.keras.models.load_model(keras_path)
        pred_ids = model.predict(X.toarray(), verbose=0).argmax(axis=1)
    else:
        pred_ids = bundle["model"].predict(X)

    labels = bundle["label_encoder"].inverse_transform(pred_ids)
    return labels.tolist()


if __name__ == "__main__":
    samples = [
        "tri_tue_nhan_tao hoc_may cong_nghe so",
        "iphone samsung man_hinh camera pin",
    ]
    for text, label in zip(samples, predict_texts(samples)):
        print(f"{label}: {text}")

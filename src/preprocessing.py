import pandas as pd
import numpy as np
from underthesea import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import os
import joblib

class TextPreprocessor:
    def __init__(self, stopword_path=None, max_features=None):
        self.stopword_path = stopword_path
        self.stopwords = self._load_stopwords()
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.svd = None

    def _load_stopwords(self):
        """Tải danh sách từ dừng."""
        if self.stopword_path and os.path.exists(self.stopword_path):
            try:
                with open(self.stopword_path, 'r', encoding='utf-8') as f:
                    return set([line.strip() for line in f])
            except Exception as e:
                print(f"Lỗi khi tải danh sách từ dừng: {e}")
        return set()

    def process_text(self, text):
        """Tiền xử lý văn bản: chuyển chữ thường, tách từ và loại bỏ từ dừng."""
        if not isinstance(text, str) or text.strip() == "":
            return ""
        
        # 1. Chuyển văn bản về chữ thường
        text = text.lower()
        
        # 2. Tách từ tiếng Việt
        tokens = word_tokenize(text, format="text")
        
        # 3. Loại bỏ từ dừng
        words = tokens.split()
        filtered_words = [w for w in words if w not in self.stopwords]
        
        return " ".join(filtered_words)

    def fit_transform_tfidf(self, corpus):
        """Biến đổi tập văn bản thành ma trận TF-IDF"""
        print("--- Đang trích xuất đặc trưng TF-IDF ---")
        return self.vectorizer.fit_transform(corpus)

    def reduce_dimension(self, X, n_components=100):
        """Giảm chiều dữ liệu bằng TruncatedSVD (Latent Semantic Analysis)."""
        if X.shape[1] <= n_components:
            print(f"Số lượng đặc trưng ({X.shape[1]}) nhỏ hơn n_components ({n_components}). Không cần giảm chiều.")
            return X
        
        print(f"--- Đang giảm chiều dữ liệu từ {X.shape[1]} xuống {n_components} thành phần ---")
        self.svd = TruncatedSVD(n_components=n_components)
        return self.svd.fit_transform(X)

    def save_features(self, X, matrix_path, vectorizer_path):
        """Lưu ma trận đặc trưng và vectorizer cho các bước tiếp theo."""
        matrix_dir = os.path.dirname(matrix_path)
        vectorizer_dir = os.path.dirname(vectorizer_path)
        if matrix_dir:
            os.makedirs(matrix_dir, exist_ok=True)
        if vectorizer_dir:
            os.makedirs(vectorizer_dir, exist_ok=True)
        print(f"--- Đang lưu ma trận đặc trưng vào {matrix_path} ---")
        joblib.dump(X, matrix_path)
        print(f"--- Đang lưu vectorizer vào {vectorizer_path} ---")
        joblib.dump(self.vectorizer, vectorizer_path)


def preprocess_nlp(
    df,
    stopword_path=None,
    text_column="main_content",
    processed_column="processed_content",
    output_matrix_path=None,
    output_vectorizer_path=None,
    n_components=None,
    max_features=None,
):
    """
    Tiền xử lý NLP cho DataFrame và tạo đặc trưng TF-IDF.

    Args:
        df (pd.DataFrame): Dữ liệu đầu vào.
        stopword_path (str): Đường dẫn file stopwords tiếng Việt.
        text_column (str): Cột văn bản cần xử lý.
        processed_column (str): Cột lưu văn bản đã xử lý.
        output_matrix_path (str): File lưu ma trận TF-IDF.
        output_vectorizer_path (str): File lưu vectorizer.
        n_components (int): Số chiều sau khi giảm chiều bằng TruncatedSVD.
        max_features (int): Số đặc trưng TF-IDF tối đa.

    Returns:
        tuple[pd.DataFrame, scipy.sparse matrix | np.ndarray]: DataFrame đã thêm
        cột xử lý và ma trận đặc trưng.
    """
    if text_column not in df.columns:
        raise ValueError(f"Không tìm thấy cột '{text_column}' trong dữ liệu.")

    processor = TextPreprocessor(
        stopword_path=stopword_path,
        max_features=max_features,
    )

    df_processed = df.copy()
    df_processed[processed_column] = (
        df_processed[text_column]
        .fillna("")
        .apply(processor.process_text)
    )

    X_features = processor.fit_transform_tfidf(df_processed[processed_column])

    if n_components:
        X_features = processor.reduce_dimension(X_features, n_components=n_components)

    if output_matrix_path and output_vectorizer_path:
        processor.save_features(X_features, output_matrix_path, output_vectorizer_path)

    return df_processed, X_features


# --- Chạy quy trình tiền xử lý dữ liệu ---
if __name__ == "__main__":
    # Cấu hình đường dẫn đầu vào và đầu ra
    DATA_PATH = "data/cleaned_news.csv"
    STOPWORD_PATH = "data/vietnamese-stopwords.txt"
    OUTPUT_MATRIX = "data/tfidf_features.pkl"
    OUTPUT_VECTORIZER = "models/vectorizer.pkl"
    OUTPUT_PROCESSED = "data/processed_news.csv"
    
    # Tạo thư mục models nếu chưa có
    if not os.path.exists("models"):
        os.makedirs("models")

    # 1. Đọc dữ liệu đã làm sạch
    if os.path.exists(DATA_PATH):
        print(f"--- Đang đọc dữ liệu từ {DATA_PATH} ---")
        df = pd.read_csv(DATA_PATH)
        
        # Kiểm tra cột main_content
        if 'main_content' in df.columns:
            print("--- Đang tiền xử lý nội dung văn bản (có thể mất chút thời gian)... ---")
            df_subset, X_tfidf = preprocess_nlp(
                df,
                stopword_path=STOPWORD_PATH,
                text_column="main_content",
                processed_column="processed_content",
                output_matrix_path=OUTPUT_MATRIX,
                output_vectorizer_path=OUTPUT_VECTORIZER,
            )
            print(f"Ma trận TF-IDF hoàn tất: {X_tfidf.shape}")
            print(f"--- Đang lưu dữ liệu đã tiền xử lý vào {OUTPUT_PROCESSED} ---")
            df_subset.to_csv(OUTPUT_PROCESSED, index=False, encoding='utf-8-sig')

            print("\n--- HOÀN TẤT! Dữ liệu đã được tiền xử lý và lưu lại. ---")
        else:
            print("Lỗi: Không tìm thấy cột 'main_content' trong tệp dữ liệu.")
    else:
        print(f"Lỗi: Không tìm thấy tệp {DATA_PATH}. Vui lòng kiểm tra lại đường dẫn.")

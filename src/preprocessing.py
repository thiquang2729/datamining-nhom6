import pandas as pd
import numpy as np
from underthesea import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import os
import joblib

class TextPreprocessor:
    def __init__(self, stopword_path=None):
        self.stopword_path = stopword_path
        self.stopwords = self._load_stopwords()
        self.vectorizer = TfidfVectorizer()
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
        print(f"--- Đang lưu ma trận đặc trưng vào {matrix_path} ---")
        joblib.dump(X, matrix_path)
        print(f"--- Đang lưu vectorizer vào {vectorizer_path} ---")
        joblib.dump(self.vectorizer, vectorizer_path)

# --- Chạy quy trình tiền xử lý dữ liệu ---
if __name__ == "__main__":
    # Cấu hình đường dẫn đầu vào và đầu ra
    DATA_PATH = "data/cleaned_news.csv"
    STOPWORD_PATH = "data/vietnamese-stopwords.txt"
    OUTPUT_MATRIX = "data/tfidf_features.pkl"
    OUTPUT_VECTORIZER = "models/vectorizer.pkl"
    
    # Tạo thư mục models nếu chưa có
    if not os.path.exists("models"):
        os.makedirs("models")

    # Khởi tạo bộ tiền xử lý với danh sách từ dừng
    print(f"--- Khởi tạo Preprocessor với {STOPWORD_PATH} ---")
    processor = TextPreprocessor(stopword_path=STOPWORD_PATH)
    
    # 1. Đọc dữ liệu đã làm sạch
    if os.path.exists(DATA_PATH):
        print(f"--- Đang đọc dữ liệu từ {DATA_PATH} ---")
        df = pd.read_csv(DATA_PATH)
        
        # Kiểm tra cột main_content
        if 'main_content' in df.columns:
            print("--- Đang tiền xử lý nội dung văn bản (có thể mất chút thời gian)... ---")
            # Sao chép dữ liệu để giữ nguyên DataFrame gốc
            df_subset = df.copy()
            df_subset['processed_content'] = df_subset['main_content'].apply(processor.process_text)
            
            # 2. Trích xuất đặc trưng TF-IDF
            X_tfidf = processor.fit_transform_tfidf(df_subset['processed_content'])
            print(f"Ma trận TF-IDF hoàn tất: {X_tfidf.shape}")
            
            # 3. Lưu kết quả để sử dụng ở các bước tiếp theo
            processor.save_features(X_tfidf, OUTPUT_MATRIX, OUTPUT_VECTORIZER)
            
            print("\n--- HOÀN TẤT! Dữ liệu đã được tiền xử lý và lưu lại. ---")
        else:
            print("Lỗi: Không tìm thấy cột 'main_content' trong tệp dữ liệu.")
    else:
        print(f"Lỗi: Không tìm thấy tệp {DATA_PATH}. Vui lòng kiểm tra lại đường dẫn.")

import pandas as pd
import numpy as np
from underthesea import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import os

class TextPreprocessor:
    def __init__(self, stopword_path=None):
        self.stopword_path = stopword_path
        self.stopwords = self._load_stopwords()
        self.vectorizer = TfidfVectorizer()
        self.svd = None

    def _load_stopwords(self):
        """Tải danh sách từ dừng (Stopwords)"""
        if self.stopword_path and os.path.exists(self.stopword_path):
            try:
                with open(self.stopword_path, 'r', encoding='utf-8') as f:
                    return set([line.strip() for line in f])
            except Exception as e:
                print(f"Lỗi khi tải stopwords: {e}")
        return set()

    def process_text(self, text):
        """Tiền xử lý văn bản: Lowercase -> Tokenize -> Remove Stopwords"""
        if not isinstance(text, str):
            return ""
        
        # 1. Lowercase
        text = text.lower()
        
        # 2. Word Tokenization (Tách từ tiếng Việt)
        # Sử dụng format="text" để nối các từ phức bằng dấu gạch dưới (ví dụ: trí_tuệ_nhân_tạo)
        tokens = word_tokenize(text, format="text")
        
        # 3. Lọc Stopwords
        words = tokens.split()
        filtered_words = [w for w in words if w not in self.stopwords]
        
        return " ".join(filtered_words)

    def fit_transform_tfidf(self, corpus):
        """Biến đổi tập văn bản thành ma trận TF-IDF"""
        print("--- Đang trích xuất đặc trưng TF-IDF ---")
        return self.vectorizer.fit_transform(corpus)

    def reduce_dimension(self, X, n_components=100):
        """Giảm chiều dữ liệu bằng TruncatedSVD (Latent Semantic Analysis)"""
        if X.shape[1] <= n_components:
            print(f"Số lượng đặc trưng ({X.shape[1]}) nhỏ hơn n_components ({n_components}). Không cần giảm chiều.")
            return X
        
        print(f"--- Đang giảm chiều dữ liệu từ {X.shape[1]} xuống {n_components} components ---")
        self.svd = TruncatedSVD(n_components=n_components)
        return self.svd.fit_transform(X)

# --- Test area cho Đại ca Nhớ ---
if __name__ == "__main__":
    # Dữ liệu mẫu để test
    sample_docs = [
        "Nhớ đang sử dụng Linux Mint để làm bài tập Data Mining.",
        "Trí tuệ nhân tạo và học máy là những lĩnh vực rất thú vị.",
        "Việc tách từ tiếng Việt giúp máy tính hiểu văn bản tốt hơn."
    ]
    
    # Khởi tạo preprocessor
    processor = TextPreprocessor(stopword_path="data/vietnamese-stopwords.txt")
    
    # 1. Chạy tiền xử lý
    print("\n[1] Đang tiền xử lý văn bản...")
    cleaned_docs = [processor.process_text(doc) for doc in sample_docs]
    for i, doc in enumerate(cleaned_docs):
        print(f"Doc {i+1}: {doc}")
        
    # 2. Chạy TF-IDF
    print("\n[2] Đang tạo ma trận TF-IDF...")
    tfidf_matrix = processor.fit_transform_tfidf(cleaned_docs)
    print(f"Ma trận TF-IDF có kích thước: {tfidf_matrix.shape}")
    
    # 3. Giảm chiều (Test với n_components=2 vì dữ liệu mẫu ít)
    print("\n[3] Đang thử nghiệm giảm chiều...")
    reduced_matrix = processor.reduce_dimension(tfidf_matrix, n_components=2)
    print(f"Kích thước sau khi giảm chiều: {reduced_matrix.shape}")
    
    print("\n--- Hoàn tất chạy thử! Đại ca thấy mượt không? ---")

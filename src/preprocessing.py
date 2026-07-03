import pandas as pd
import numpy as np
from underthesea import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import os
import joblib
from tqdm import tqdm

tqdm.pandas()

class TextPreprocessor:
    def __init__(self, stopword_path=None, max_features=None):
        self.stopword_path = stopword_path
        self.stopwords = self._load_stopwords()
        self.vectorizer = TfidfVectorizer(
            max_features=max_features or 10000,
            min_df=5,
            max_df=0.7
        )
        self.svd = None

    def _load_stopwords(self):
        """Tải danh sách từ dừng."""
        if self.stopword_path and os.path.exists(self.stopword_path):
            try:
                with open(self.stopword_path, 'r', encoding='utf-8') as f:
                    stopwords = set()
                    for line in f:
                        word = line.strip().lower()
                        if not word or word.startswith("#"):
                            continue
                        stopwords.add(word)
                        stopwords.add(word.replace(" ", "_"))
                    return stopwords
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
        filtered_words = [
            w for w in words
            if w not in self.stopwords
            and len(w) > 1  # loại ký tự đơn lẻ
        ]
        
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


def encode_labels(df, label_column='category', encoding_method='label'):
    """
    Mã hóa cột nhãn phân loại.
    Args:
        df (pd.DataFrame): DataFrame đầu vào.
        label_column (str): Tên cột cần mã hóa.
        encoding_method (str): 'label' cho LabelEncoder, 'onehot' cho OneHotEncoder.
    Returns:
        tuple: (df đã mã hóa, encoder đã fit)
    """
    if label_column not in df.columns:
        print(f"Không tìm thấy cột '{label_column}' để mã hóa.")
        return df, None
    
    df_encoded = df.copy()
    if encoding_method == 'label':
        le = LabelEncoder()
        df_encoded[f'{label_column}_encoded'] = le.fit_transform(df_encoded[label_column].astype(str))
        return df_encoded, le
    elif encoding_method == 'onehot':
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded = ohe.fit_transform(df_encoded[[label_column]])
        feature_names = ohe.get_feature_names_out([label_column])
        df_encoded = pd.concat([df_encoded.reset_index(drop=True), pd.DataFrame(encoded, columns=feature_names)], axis=1)
        return df_encoded, ohe
    return df, None


def encode_source(df, source_column='source'):
    """
    Mã hóa cột nguồn bằng OneHotEncoder.
    Args:
        df (pd.DataFrame): DataFrame đầu vào.
        source_column (str): Tên cột nguồn.
    Returns:
        tuple: (df đã mã hóa, OneHotEncoder đã fit)
    """
    if source_column not in df.columns:
        print(f"Không tìm thấy cột '{source_column}' để mã hóa.")
        return df, None
    
    df_encoded = df.copy()
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded = ohe.fit_transform(df_encoded[[source_column]])
    feature_names = ohe.get_feature_names_out([source_column])
    df_encoded = pd.concat([df_encoded.reset_index(drop=True), pd.DataFrame(encoded, columns=feature_names)], axis=1)
    return df_encoded, ohe


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
        .progress_apply(processor.process_text)
    )

    X_features = processor.fit_transform_tfidf(df_processed[processed_column])

    # Vẽ các biểu đồ báo cáo trước khi thực hiện giảm chiều dữ liệu
    visualize_preprocessing_report(
        df_processed,
        X_features,
        processor.vectorizer,
        text_column=text_column,
        processed_column=processed_column,
    )

    if n_components:
        X_features = processor.reduce_dimension(X_features, n_components=n_components)

    if output_matrix_path and output_vectorizer_path:
        processor.save_features(X_features, output_matrix_path, output_vectorizer_path)

    return df_processed, X_features


def visualize_preprocessing_report(
    df_processed,
    X_features,
    vectorizer,
    text_column="main_content",
    processed_column="processed_content",
    save_dir=None,
    top_n=20,
):
    """
    Vẽ biểu đồ trực quan hóa bước tiền xử lý NLP và TF-IDF.
    Args:
        df_processed (pd.DataFrame): Dữ liệu đã có cột processed_content.
        X_features: Ma trận đặc trưng TF-IDF.
        vectorizer: TfidfVectorizer đã fit.
        text_column (str): Cột văn bản gốc.
        processed_column (str): Cột văn bản sau tiền xử lý.
        save_dir (str): Thư mục lưu ảnh. Mặc định: notebooks/.
        top_n (int): Số từ/cụm từ TF-IDF quan trọng nhất cần hiển thị.

    Returns:
        list[str]: Danh sách đường dẫn ảnh đã tạo.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    if text_column not in df_processed.columns:
        raise ValueError(f"Không tìm thấy cột '{text_column}' trong dữ liệu.")
    if processed_column not in df_processed.columns:
        raise ValueError(f"Không tìm thấy cột '{processed_column}' trong dữ liệu.")

    if save_dir is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        save_dir = os.path.join(project_root, 'notebooks')

    os.makedirs(save_dir, exist_ok=True)

    plt.rcParams['font.family'] = 'DejaVu Sans'
    sns.set_theme(style="whitegrid", palette="muted")

    original_word_counts = (
        df_processed[text_column]
        .fillna("")
        .astype(str)
        .apply(lambda text: len(text.split()))
    )
    processed_word_counts = (
        df_processed[processed_column]
        .fillna("")
        .astype(str)
        .apply(lambda text: len(text.split()))
    )
    removed_word_counts = (original_word_counts - processed_word_counts).clip(lower=0)

    created_paths = []

    # BIỂU ĐỒ 1: So sánh độ dài văn bản trước/sau tiền xử lý
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    fig1.suptitle('So sánh độ dài văn bản trước/sau tiền xử lý NLP',
                  fontsize=14, fontweight='bold')

    sns.histplot(
        original_word_counts,
        bins=40,
        color='#4C78A8',
        label='Trước xử lý',
        alpha=0.45,
        ax=ax1,
    )
    sns.histplot(
        processed_word_counts,
        bins=40,
        color='#F58518',
        label='Sau xử lý',
        alpha=0.45,
        ax=ax1,
    )
    ax1.axvline(
        original_word_counts.mean(),
        color='#4C78A8',
        linestyle='--',
        linewidth=1.5,
    )
    ax1.axvline(
        processed_word_counts.mean(),
        color='#F58518',
        linestyle='--',
        linewidth=1.5,
    )
    ax1.set_xlabel('Số từ trong mỗi bài viết', fontsize=12)
    ax1.set_ylabel('Số bài viết', fontsize=12)
    ax1.legend(
        title=f'TB: {original_word_counts.mean():.1f} → '
              f'{processed_word_counts.mean():.1f}'
    )

    path1 = os.path.join(save_dir, 'preprocessing_01_word_count_distribution.png')
    plt.tight_layout()
    fig1.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    created_paths.append(path1)

    # BIỂU ĐỒ 2: Trung bình số từ trước/sau và số từ đã loại
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    fig2.suptitle('Tác động của tiền xử lý lên số lượng từ',
                  fontsize=14, fontweight='bold')

    labels = ['Trước xử lý', 'Sau xử lý', 'Từ đã loại']
    values = [
        original_word_counts.mean(),
        processed_word_counts.mean(),
        removed_word_counts.mean(),
    ]
    colors = ['#4C78A8', '#54A24B', '#E45756']

    bars = ax2.bar(labels, values, color=colors, edgecolor='white', linewidth=1.5)
    ax2.set_ylabel('Số từ trung bình / bài viết', fontsize=12)

    for bar, value in zip(bars, values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f'{value:.1f}',
            ha='center',
            va='bottom',
            fontsize=12,
            fontweight='bold',
        )

    path2 = os.path.join(save_dir, 'preprocessing_02_word_reduction_bar.png')
    plt.tight_layout()
    fig2.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    created_paths.append(path2)

    # BIỂU ĐỒ 3: Top từ/cụm từ có điểm TF-IDF trung bình cao nhất
    feature_names = vectorizer.get_feature_names_out()
    mean_tfidf = np.asarray(X_features.mean(axis=0)).ravel()

    if len(feature_names) != len(mean_tfidf):
        print("Không vẽ biểu đồ top TF-IDF vì ma trận đã được giảm chiều.")
        return created_paths

    top_indexes = mean_tfidf.argsort()[-top_n:][::-1]
    top_terms = feature_names[top_indexes]
    top_scores = mean_tfidf[top_indexes]

    fig3, ax3 = plt.subplots(figsize=(10, 7))
    fig3.suptitle(f'Top {top_n} từ/cụm từ nổi bật theo TF-IDF',
                  fontsize=14, fontweight='bold')

    sns.barplot(x=top_scores, y=top_terms, ax=ax3, color='#72B7B2')
    ax3.set_xlabel('Điểm TF-IDF trung bình', fontsize=12)
    ax3.set_ylabel('Từ/cụm từ', fontsize=12)

    path3 = os.path.join(save_dir, 'preprocessing_03_top_tfidf_terms.png')
    plt.tight_layout()
    fig3.savefig(path3, dpi=150, bbox_inches='tight')
    plt.close(fig3)
    created_paths.append(path3)

    print("--- Đã tạo biểu đồ trực quan hóa bước NLP/TF-IDF ---")
    for path in created_paths:
        print(path)

    return created_paths


# --- Chạy quy trình tiền xử lý dữ liệu ---
if __name__ == "__main__":
    DATA_PATH = "data/cleaned_news.csv"
    STOPWORD_PATH = "data/vietnamese-stopwords.txt"
    OUTPUT_MATRIX = "data/tfidf_features.pkl"
    OUTPUT_VECTORIZER = "models/vectorizer.pkl"
    OUTPUT_PROCESSED = "data/processed_news.csv"
    
    if not os.path.exists("models"):
        os.makedirs("models")

    if os.path.exists(DATA_PATH):
        print(f"--- Đang đọc dữ liệu từ {DATA_PATH} ---")
        df = pd.read_csv(DATA_PATH)
        
        if 'main_content' in df.columns:
            print("--- Đang tiền xử lý nội dung văn bản (có thể mất chút thời gian)... ---")
            df_subset, X_tfidf = preprocess_nlp(
                df,
                stopword_path=STOPWORD_PATH,
                text_column="main_content",
                processed_column="processed_content",
                output_matrix_path=OUTPUT_MATRIX,
                output_vectorizer_path=OUTPUT_VECTORIZER,
                n_components=300,
                max_features=3000,
            )
            print(f"Ma trận TF-IDF hoàn tất: {X_tfidf.shape}")
            print(f"--- Đang lưu dữ liệu đã tiền xử lý vào {OUTPUT_PROCESSED} ---")
            df_subset.to_csv(OUTPUT_PROCESSED, index=False, encoding='utf-8-sig')
            print("\n--- HOÀN TẤT! Dữ liệu đã được tiền xử lý và lưu lại. ---")
        else:
            print("Lỗi: Không tìm thấy cột 'main_content' trong tệp dữ liệu.")
    else:
        print(f"Lỗi: Không tìm thấy tệp {DATA_PATH}. Vui lòng kiểm tra lại đường dẫn.")
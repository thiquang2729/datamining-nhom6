# CODEBASE — Tài liệu Chi tiết Dự án Data Mining Tin tức Công nghệ

> **Nhóm 6 — Data Mining**
> Dự án xây dựng pipeline xử lý dữ liệu tin tức công nghệ: từ thu thập, làm sạch, chuẩn hóa, loại trùng lặp, tiền xử lý NLP, phân cụm, gán nhãn đến huấn luyện mô hình phân loại.

---

## Mục lục

1. [Tổng quan hệ thống](#1-tổng-quan-hệ-thống)
2. [Workflow toàn bộ Pipeline](#2-workflow-toàn-bộ-pipeline)
3. [Cấu trúc thư mục](#3-cấu-trúc-thư-mục)
4. [Thư viện sử dụng và tác dụng](#4-thư-viện-sử-dụng-và-tác-dụng)
5. [Chi tiết từng Module](#5-chi-tiết-từng-module)
   - 5.1 [config.json — Cấu hình Pipeline](#51-configjson--cấu-hình-pipeline)
   - 5.2 [logger.py — Logging có màu](#52-loggerpy--logging-có-màu)
   - 5.3 [integration.py — Tích hợp dữ liệu](#53-integrationpy--tích-hợp-dữ-liệu)
   - 5.4 [cleaning.py — Làm sạch dữ liệu](#54-cleaningpy--làm-sạch-dữ-liệu)
   - 5.5 [normalizer.py — Chuẩn hóa văn bản](#55-normalizerpy--chuẩn-hóa-văn-bản)
   - 5.6 [deduplicator.py — Loại trùng lặp](#56-deduplicatorpy--loại-trùng-lặp)
   - 5.7 [preprocessing.py — Tiền xử lý NLP & TF-IDF](#57-preprocessingpy--tiền-xử-lý-nlp--tf-idf)
   - 5.8 [modeling.py — Phân cụm & Gán nhãn](#58-modelingpy--phân-cụm--gán-nhãn)
   - 5.9 [deep_learning_model.py — Huấn luyện mô hình Deep Learning](#59-deep_learning_modelpy--huấn-luyện-mô-hình-deep-learning)
   - 5.10 [predict.py — Dự đoán dữ liệu mới](#510-predictpy--dự-đoán-dữ-liệu-mới)
   - 5.11 [pipeline.py — Điều phối Pipeline](#511-pipelinepy--điều-phối-pipeline)
   - 5.12 [visualize_collection.py — Trực quan hóa EDA](#512-visualize_collectionpy--trực-quan-hóa-eda)
   - 5.13 [visualize_extra.py — Biểu đồ bổ sung](#513-visualize_extrapy--biểu-đồ-bổ-sung)
6. [Luồng dữ liệu giữa các file](#6-luồng-dữ-liệu-giữa-các-file)
7. [Phân công Thành viên](#7-phân-công-thành-viên)
8. [Cách chạy](#8-cách-chạy)

---

## 1. Tổng quan hệ thống

Đây là dự án **Data Mining** xử lý tập dữ liệu tin tức công nghệ tiếng Việt, thu thập từ 3 nguồn báo (VnExpress, VietnamNet, Thanh Niên). Hệ thống hoạt động như một **pipeline tuần tự**, mỗi bước nhận đầu vào từ bước trước và tạo đầu ra cho bước sau.

**Mục tiêu cuối cùng**: Từ dữ liệu thô (bài báo) → xử lý → phân cụm tự động → gán nhãn chủ đề → huấn luyện mô hình phân loại văn bản → dự đoán chủ đề cho bài báo mới.

**Ngôn ngữ**: Python 3

**Kiến trúc**: Pipeline tuần tự, mỗi module là một file `.py` riêng biệt, được kết nối qua `pipeline.py` (file điều phối chính).

---

## 2. Workflow toàn bộ Pipeline

### 2.1 Sơ đồ tổng thể (từ đầu đến cuối)

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PIPELINE XỬ LÝ DỮ LIỆU TIN TỨC                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  data/vnexpress.csv, thanhnien.csv, vietnamnet.csv (Dữ liệu nguồn)          │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─── Bước 1: Tích hợp dữ liệu (Integration) ─ [integration.py]             │
│  │    Ghép 3 nguồn → Lọc giữ 7 cột chính → Lưu data/news_data.csv            │
│  │                                                                          │
│  ├─── Bước 2: Làm sạch dữ liệu ─────────────── [cleaning.py]             │
│  │    Loại dòng rỗng → Lọc độ dài bất thường                                 │
│  │                                                                          │
│  ├─── Bước 3: Chuẩn hóa văn bản ─────────────── [normalizer.py]           │
│  │    Unicode NFC → Xóa HTML → Xóa emoji → Xóa ký tự đặc biệt           │
│  │    → Chuẩn hóa khoảng trắng                                               │
│  │                                                                          │
│  ├─── Bước 4: Loại trùng lặp ────────────────── [deduplicator.py]         │
│  │    So sánh trực tiếp cột main_content → Loại bản ghi trùng             │
│  │                                                                          │
│  │    =====> Lưu: data/cleaned_news.csv                                    │
│  │                                                                          │
│  ├─── Bước 5: Tiền xử lý NLP ────────────────── [preprocessing.py]        │
│  │    Chuyển lowercase → Tách từ tiếng Việt (underthesea)                  │
│  │    → Loại stopwords → Tạo TF-IDF → Giảm chiều TruncatedSVD            │
│  │                                                                          │
│  │    =====> Lưu: data/processed_news.csv                                  │
│  │    =====> Lưu: data/tfidf_features.pkl                                  │
│  │    =====> Lưu: models/vectorizer.pkl                                    │
│  │                                                                          │
│  ├─── Bước 6: Mã hóa nhãn ───────────────────── [preprocessing.py]        │
│  │    Label Encoding cho cột category                                      │
│  │                                                                          │
│  ├─── Bước 7: Mã hóa nguồn ──────────────────── [preprocessing.py]        │
│  │    One-Hot Encoding cho cột source                                      │
│  │                                                                          │
│  ├─── Bước 8: Phân cụm & Gán nhãn ───────────── [modeling.py]             │
│  │    KMeans (k=5) → Gán nhãn chủ đề → DBSCAN (so sánh)                   │
│  │    → Chia train/val/test (70/15/15)                                     │
│  │                                                                          │
│  │    =====> Lưu: data/labeled_news.csv                                    │
│  │    =====> Lưu: data/train.csv, val.csv, test.csv                        │
│  │    =====> Lưu: models/kmeans_model.pkl                                  │
│  │                                                                          │
│  └─── Bước 9: Huấn luyện mô hình ────────────── [deep_learning_model.py]  │
│       MLP (sklearn) hoặc Dense (TensorFlow)                                │
│       → Đánh giá Accuracy, F1 → Xuất biểu đồ                              │
│                                                                             │
│       =====> Lưu: models/deep_learning_text_model.pkl                      │
│       =====> Lưu: models/deep_learning_metrics.json                        │
│                                                                             │
│  ┌─── Dự đoán ──────────────────────────────────  [predict.py]             │
│  │    Nạp model đã train → Dự đoán nhãn cho văn bản mới                   │
│  └──────────────────────────────────────────────────────────────            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Tóm tắt luồng dữ liệu

```text
                                                    ┌─────────────────┼────────────────┐
                                                    ▼                 ▼                ▼
                                          processed_news.csv  tfidf_features.pkl  vectorizer.pkl
                                                                      │
                                                                      ▼
                                                              modeling (KMeans)
                                                                      │
                                                    ┌─────────────────┼────────────────┐
                                                    ▼                 ▼                ▼
                                               train.csv          val.csv          test.csv
                                                    │                 │                │
                                                    └────────────┬────┘                │
                                                                 ▼                     │
                                                        deep_learning_model            │
                                                                 │                     │
                                                                 ▼                     │
                                                    deep_learning_text_model.pkl       │
                                                                 │                     │
                                                                 ▼                     │
                                                            predict.py ←───────────────┘
```

---

## 3. Cấu trúc thư mục

```text
Nhom8_DataMining_NEWS/
│
├── config.json                    # Cấu hình tham số cho toàn bộ pipeline
├── requirements.txt               # Danh sách thư viện Python cần cài
├── README.md                      # Hướng dẫn sử dụng tổng quan
├── CODEBASE.md                    # ← File này: tài liệu chi tiết codebase
│
├── src/                           # ★ THƯ MỤC MÃ NGUỒN CHÍNH
│   ├── pipeline.py                # Điều phối toàn bộ pipeline (file entry point)
│   ├── integration.py             # Module tích hợp dữ liệu từ 3 nguồn báo
│   ├── cleaning.py                # Module làm sạch dữ liệu
│   ├── normalizer.py              # Module chuẩn hóa văn bản
│   ├── deduplicator.py            # Module loại bỏ trùng lặp
│   ├── preprocessing.py           # Module tiền xử lý NLP & TF-IDF
│   ├── modeling.py                # Module phân cụm KMeans/DBSCAN & gán nhãn
│   ├── deep_learning_model.py     # Module huấn luyện mô hình phân loại
│   ├── predict.py                 # Module dự đoán dữ liệu mới
│   ├── logger.py                  # Module logging có màu sắc trên terminal
│   ├── visualize_collection.py    # Trực quan hóa EDA (dữ liệu thu thập)
│   └── visualize_extra.py         # Biểu đồ bổ sung (correlation, scatter)
│
├── data/                          # ★ THƯ MỤC DỮ LIỆU
│   ├── news_data.csv              # Dữ liệu thô (không đẩy lên GitHub nếu quá lớn)
│   ├── cleaned_news.csv           # Dữ liệu sau bước làm sạch/chuẩn hóa/loại trùng
│   ├── processed_news.csv         # Dữ liệu sau tiền xử lý NLP (có cột processed_content)
│   ├── tfidf_features.pkl         # Ma trận đặc trưng TF-IDF (dạng sparse/dense)
│   ├── vietnamese-stopwords.txt   # Danh sách 113 từ dừng tiếng Việt
│   ├── checkpoints/               # Checkpoint lưu sau mỗi bước pipeline
│   │   └── 05_preprocessed.csv    # Checkpoint sau bước tiền xử lý NLP
│   └── deep_learning_confusion_matrix.csv  # Ma trận nhầm lẫn của mô hình DL
│
├── models/                        # ★ THƯ MỤC MÔ HÌNH ĐÃ TRAIN
│   ├── vectorizer.pkl             # TF-IDF vectorizer đã fit (dùng cho predict)
│   ├── kmeans_model.pkl           # Mô hình KMeans đã train
│   ├── deep_learning_text_model.pkl  # Mô hình phân loại đã train (MLP hoặc TF)
│   └── deep_learning_metrics.json # Kết quả đánh giá mô hình DL
│
├── notebooks/                     # ★ THƯ MỤC BIỂU ĐỒ & BÁO CÁO
│   ├── notebook.ipynb             # Notebook (placeholder)
│   │
│   │   # --- Biểu đồ bước Làm sạch (cleaning.py) ---
│   ├── cleaning_01_error_types.png          # Bar chart các loại lỗi đã loại bỏ
│   ├── cleaning_04_scatter_wordcount.png    # Scatter plot word count sau cleaning
│   │
│   │   # --- Biểu đồ bước Chuẩn hóa (normalizer.py) ---
│   ├── normalize_01_noise_types.png         # Bar chart loại nhiễu đã loại
│   ├── normalize_02_chars_comparison.png    # So sánh ký tự trước/sau chuẩn hóa
│   ├── normalize_03_noise_distribution.png  # Pie chart tỷ lệ nhiễu
│   │
│   │   # --- Biểu đồ bước Loại trùng (deduplicator.py) ---
│   ├── dedup_01_domain_bar.png              # Bar chart theo domain trước/sau
│   │
│   │   # --- Biểu đồ bước Phân cụm (modeling.py) ---
│   ├── elbow_method.png                     # Đồ thị Elbow tìm K tối ưu
│   ├── cluster_distribution.png             # Phân bố bài theo cụm KMeans
│   ├── cluster_distance.png                 # Khoảng cách trung bình đến centroid
│   ├── dbscan_distribution.png              # Phân bố cụm DBSCAN
│   ├── kmeans_vs_dbscan.png                 # So sánh số cụm KMeans vs DBSCAN
│   ├── pca_cluster.png                      # PCA 2D visualization
│   ├── silhouette_plot.png                  # Silhouette analysis
│   ├── kmeans_center_heatmap.png            # Heatmap cluster centers
│   ├── top20_tfidf.png                      # Top 20 từ TF-IDF
│   ├── tfidf_heatmap.png                    # Heatmap ma trận TF-IDF
│   ├── wordcloud_cluster_*.png              # Word cloud cho từng cụm
│   ├── sample_label_distribution.png        # Phân bố mẫu gán nhãn tay
│   ├── dataset_split_pie.png                # Pie chart chia train/val/test
│   ├── dataset_splitting.png                # Biểu đồ chia dataset
│   │
│   │   # --- Biểu đồ bước Deep Learning (deep_learning_model.py) ---
│   ├── deep_learning_01_confusion_matrix.png  # Confusion matrix
│   ├── deep_learning_02_metrics.png           # Accuracy/F1 bar chart
│   ├── deep_learning_03_class_f1.png          # F1-score theo từng lớp
│   ├── deep_learning_04_dataset_split.png     # Train/Val/Test size
│   │
│   │   # --- Biểu đồ Pipeline tổng (pipeline.py) ---
│   ├── pipeline_funnel.png                  # Waterfall số bản ghi qua từng bước
│   │
│   │   # --- Biểu đồ EDA Data Collection (visualize_collection.py) ---
│   ├── collection_01_metric_cards.png       # 4 metric cards tổng quan
│   ├── collection_02_domain_donut.png       # Donut nguồn báo
│   ├── collection_03_subcategory.png        # Bar sub-category
│   ├── collection_04_timeline.png           # Timeline thu thập theo tháng
│   ├── collection_05_cluster_donut.png      # Donut nhãn K-Means
│   ├── collection_06_top_tags.png           # Top tags nổi bật
│   ├── collection_07_content_length.png     # Histogram độ dài nội dung
│   ├── collection_08_dashboard.png          # Dashboard tổng hợp 2×3
│   ├── collection_09_correlation_matrix.png # Heatmap nguồn báo vs tags
│   └── collection_10_scatter_wordcount.png  # Scatter plot word count
│
└── demo/                          # ★ THƯ MỤC DEMO
    └── app.py                     # File demo ứng dụng (placeholder)
```

### 3.1 Giải thích vai trò từng thư mục

| Thư mục | Nhiệm vụ |
|---|---|
| `src/` | Chứa toàn bộ mã nguồn Python xử lý dữ liệu. Mỗi file là một module độc lập phụ trách một bước cụ thể trong pipeline. `pipeline.py` là file trung tâm, import và gọi các module khác. |
| `data/` | Chứa dữ liệu đầu vào, dữ liệu trung gian, và các file pickle. File `.csv` là dữ liệu dạng bảng, file `.pkl` là object Python đã serialize (mô hình, ma trận). `checkpoints/` lưu trạng thái sau mỗi bước để có thể chạy lại từ giữa nếu lỗi. |
| `models/` | Chứa các mô hình đã được huấn luyện (KMeans, MLP/TF, Vectorizer) dưới dạng file `.pkl`. Khi cần dự đoán dữ liệu mới, chỉ cần nạp các file trong thư mục này mà không cần chạy lại pipeline. |
| `notebooks/` | Chứa toàn bộ biểu đồ trực quan hóa dưới dạng file `.png`. Mỗi bước trong pipeline tự động tạo biểu đồ và lưu vào đây. Dùng cho báo cáo, slide, và phân tích. |
| `demo/` | Thư mục dành cho ứng dụng demo (hiện đang là placeholder). |

---

## 4. Thư viện sử dụng và tác dụng

### 4.1 Thư viện trong `requirements.txt`

| Thư viện | Tác dụng | Dùng trong file nào |
|---|---|---|
| `pandas` | Đọc/ghi CSV, thao tác DataFrame (lọc, nhóm, transform dữ liệu bảng). Là thư viện cốt lõi cho xử lý dữ liệu dạng bảng. | Tất cả file |
| `matplotlib` | Vẽ biểu đồ (bar, pie, scatter, heatmap, line chart). Tạo các file ảnh PNG. | cleaning.py, normalizer.py, deduplicator.py, preprocessing.py, modeling.py, deep_learning_model.py, pipeline.py, visualize_*.py |
| `seaborn` | Thư viện vẽ biểu đồ cấp cao hơn matplotlib, cung cấp theme đẹp hơn và các hàm vẽ nhanh (histplot, barplot). | cleaning.py, normalizer.py, preprocessing.py |
| `colorama` | Hiển thị text có màu sắc trên terminal Windows/Linux. Giúp log dễ đọc hơn. | logger.py |
| `underthesea` | Thư viện NLP tiếng Việt. Dùng hàm `word_tokenize()` để tách từ/cụm từ tiếng Việt (ví dụ: "trí tuệ nhân tạo" → "trí_tuệ nhân_tạo"). | preprocessing.py |
| `scikit-learn` | Thư viện Machine Learning. Cung cấp: TfidfVectorizer, TruncatedSVD, KMeans, DBSCAN, MLPClassifier, LabelEncoder, train_test_split, các hàm đánh giá (silhouette_score, accuracy_score, f1_score, confusion_matrix...). | preprocessing.py, modeling.py, deep_learning_model.py |

### 4.2 Thư viện có sẵn (standard library) được sử dụng

| Thư viện | Tác dụng | Dùng trong file nào |
|---|---|---|
| `os` | Thao tác đường dẫn file, tạo thư mục, kiểm tra file tồn tại. | Tất cả file |
| `sys` | Thêm thư mục vào Python path, đọc argument dòng lệnh, cấu hình encoding. | pipeline.py, logger.py |
| `json` | Đọc/ghi file JSON (config.json, metrics). | pipeline.py, deep_learning_model.py |
| `re` | Biểu thức chính quy — dùng để tìm/xóa URL, HTML tags, emoji, ký tự đặc biệt. | cleaning.py, normalizer.py |
| `html` | Giải mã HTML entities (ví dụ: `&amp;` → `&`). | normalizer.py |
| `unicodedata` | Chuẩn hóa Unicode NFC cho tiếng Việt (đảm bảo các ký tự có dấu được biểu diễn nhất quán). | normalizer.py |
| `time` | Đo thời gian chạy mỗi bước pipeline. | pipeline.py, logger.py |
| `argparse` | Parse argument dòng lệnh (`python pipeline.py tv2`). | pipeline.py |

### 4.3 Thư viện bổ sung (không có trong requirements.txt nhưng được import)

| Thư viện | Tác dụng | Dùng trong file nào |
|---|---|---|
| `joblib` | Serialize/deserialize object Python (nhanh hơn pickle cho mảng numpy lớn). Dùng lưu/nạp ma trận TF-IDF, vectorizer, model. | preprocessing.py, modeling.py, deep_learning_model.py, predict.py |
| `numpy` | Tính toán mảng số (trung bình, sắp xếp, argsort cho top TF-IDF terms). | preprocessing.py, modeling.py, visualize_*.py |
| `tqdm` | Thanh tiến trình (progress bar) khi xử lý từng bài viết trong bước NLP. | preprocessing.py |
| `wordcloud` | Tạo hình Word Cloud cho từng cụm. | modeling.py |
| `tensorflow` | Framework Deep Learning — dùng nếu chọn backend `tensorflow_dense` thay vì `sklearn_mlp`. | deep_learning_model.py, predict.py (tùy chọn) |

---

## 5. Chi tiết từng Module

---

### 5.1 `config.json` — Cấu hình Pipeline

**Đường dẫn**: `config.json` (thư mục gốc)

**Mục đích**: Lưu toàn bộ tham số cấu hình cho pipeline, giúp thay đổi thông số mà không cần sửa code.

**Cấu trúc chi tiết**:

| Khóa | Giá trị | Ý nghĩa |
|---|---|---|
| `input_path` | `"data/news_data.csv"` | Đường dẫn file dữ liệu thô đầu vào |
| `output_cleaned_path` | `"data/cleaned_news.csv"` | Đường dẫn lưu dữ liệu sau khi làm sạch |
| `output_processed_path` | `"data/processed_news.csv"` | Đường dẫn lưu dữ liệu sau tiền xử lý NLP |
| `cleaning.required_columns` | `["title", "main_content", "url"]` | Các cột bắt buộc phải có giá trị (dòng nào thiếu sẽ bị loại) |
| `cleaning.min_words` | `50` | Bài viết dưới 50 từ bị loại (quá ngắn) |
| `cleaning.max_words` | `10000` | Bài viết trên 10.000 từ bị loại (quá dài, có thể lỗi) |
| `deduplication.column` | `"main_content"` | Cột dùng để kiểm tra trùng lặp |
| `deduplication.keep` | `"first"` | Khi trùng, giữ bản ghi xuất hiện đầu tiên |
| `preprocessing.stopword_path` | `"data/vietnamese-stopwords.txt"` | File danh sách từ dừng tiếng Việt |
| `preprocessing.content_column` | `"main_content"` | Cột văn bản gốc dùng làm đầu vào NLP |
| `preprocessing.processed_column` | `"processed_content"` | Tên cột mới chứa văn bản đã xử lý |
| `preprocessing.tfidf_matrix_path` | `"data/tfidf_features.pkl"` | Nơi lưu ma trận TF-IDF |
| `preprocessing.vectorizer_path` | `"models/vectorizer.pkl"` | Nơi lưu TF-IDF vectorizer đã fit |
| `preprocessing.n_components` | `300` | Số chiều sau khi giảm chiều bằng TruncatedSVD |
| `preprocessing.max_features` | `3000` | Số từ tối đa trong bộ từ điển TF-IDF |
| `encoding_config.category_column` | `"category"` | Cột chứa nhãn phân loại |
| `encoding_config.source_column` | `"source"` | Cột chứa nguồn báo |
| `encoding_config.label_encoding_method` | `"label"` | Phương pháp mã hóa nhãn (`"label"` hoặc `"onehot"`) |
| `encoding` | `"utf-8-sig"` | Encoding khi đọc/ghi CSV (hỗ trợ tiếng Việt, tương thích Excel) |

---

### 5.2 `logger.py` — Logging có màu

**Đường dẫn**: `src/logger.py` (127 dòng)

**Mục đích**: Cung cấp các hàm log có màu sắc trên terminal, giúp dễ theo dõi tiến trình pipeline. Được import và sử dụng bởi tất cả các module khác.

**Workflow**:

```text
Khi pipeline chạy:
  → log_step_start() in header tím khi bắt đầu mỗi bước
  → log_info() in thông tin xanh dương trong quá trình xử lý
  → log_success() in kết quả xanh lá khi bước thành công
  → log_warning() in cảnh báo vàng khi có vấn đề nhỏ
  → log_error() in lỗi đỏ khi có lỗi nghiêm trọng
  → log_step_end() in tổng kết cuối mỗi bước (trước/sau/đã loại/thời gian)
  → log_summary() in tổng kết toàn bộ pipeline
```

**Chi tiết từng hàm**:

| Hàm | Tác dụng |
|---|---|
| `log_info(message)` | In thông tin chung (màu xanh dương `[INFO]`). Dùng cho các bước trung gian, thông báo số liệu. |
| `log_success(message)` | In thông báo thành công (màu xanh lá `[SUCCESS] ✔`). Dùng khi một bước hoàn tất. |
| `log_warning(message)` | In cảnh báo (màu vàng `[WARNING] ⚠`). Dùng khi có vấn đề nhỏ không ảnh hưởng pipeline (ví dụ: thiếu cột, bước là placeholder). |
| `log_error(message)` | In lỗi nghiêm trọng (màu đỏ `[ERROR] ✘`). Dùng khi pipeline không thể tiếp tục. |
| `log_step_start(step_name, record_count)` | In header bắt đầu bước (màu tím, có dấu `=` phân cách). Hiển thị tên bước và số bản ghi đầu vào. |
| `log_step_end(step_name, before_count, after_count, elapsed_time)` | In tổng kết cuối bước: số bản ghi trước/sau, số đã loại bỏ (%), thời gian chạy. |
| `log_summary(total_before, total_after, total_time)` | In tổng kết toàn bộ pipeline với khung viền đẹp. |

**Class `StepTimer`**: Context manager (`with StepTimer(...) as timer:`) để tự động đo thời gian mỗi bước. Khi vào block, gọi `log_step_start()`; khi ra block, nếu có exception sẽ gọi `log_error()`. Property `timer.elapsed` trả về thời gian đã trôi qua.

**Thư viện sử dụng**:
- `colorama.init(autoreset=True)`: Khởi tạo hỗ trợ màu trên Windows, tự động reset màu sau mỗi print.
- `colorama.Fore`: Enum các màu foreground (CYAN, GREEN, YELLOW, RED, MAGENTA).
- `colorama.Style.RESET_ALL`: Reset toàn bộ style về mặc định.

---

### 5.3 `integration.py` — Tích hợp dữ liệu

**Đường dẫn**: `src/integration.py`

**Mục đích**: Tích hợp dữ liệu thô từ 3 file CSV nguồn báo (`vnexpress.csv`, `thanhnien.csv`, `vietnamnet.csv`) thành một file duy nhất (`news_data.csv`) và lọc chỉ giữ lại 7 cột dữ liệu quan trọng phục vụ cho các bước sau.

**Workflow chi tiết**:

```text
vnexpress.csv, thanhnien.csv, vietnamnet.csv
    │
    ▼
[1] Đọc từng file nguồn (nếu tồn tại) bằng pandas
    │
    ▼
[2] Ghép các DataFrame lại bằng pd.concat()
    │
    ▼
[3] Lọc chỉ giữ lại 7 cột quan trọng:
    article_id, title, description, main_content, tags, category, sub_category
    │
    ▼
[4] Lưu kết quả tích hợp xuống data/news_data.csv
```

**Chi tiết các hàm**:

- `integrate_data(source_paths, keep_columns, output_path, encoding)`: Nhận danh sách file nguồn, ghép, lọc cột, điền giá trị rỗng cho các cột thiếu, và lưu output.

---

### 5.4 `cleaning.py` — Làm sạch dữ liệu

**Đường dẫn**: `src/cleaning.py` (363 dòng)

**Mục đích**: Loại bỏ các bản ghi không hợp lệ khỏi tập dữ liệu thô. Đây là bước đầu tiên sau khi đọc dữ liệu, đảm bảo chỉ giữ lại các bài viết có đủ thông tin và chất lượng.

**Workflow chi tiết**:

```text
DataFrame thô (~ 20,000+ bản ghi)
    │
    ▼
[1] remove_empty_rows()
    Kiểm tra 3 cột: title, main_content, url
    Nếu cột nào rỗng (NaN, chuỗi rỗng, chỉ khoảng trắng) → loại dòng đó
    │
    ▼
[2] validate_urls()
    Kiểm tra cột url bằng regex: phải bắt đầu bằng http:// hoặc https://
    và có domain hợp lệ (ít nhất 2 ký tự TLD)
    URL không hợp lệ → loại dòng đó
    │
    ▼
[3] filter_by_length()
    Đếm số từ trong main_content
    Bài < 50 từ → loại (quá ngắn, có thể bị lỗi crawl)
    Bài > 10.000 từ → loại (có thể bị lỗi encoding/duplicate nội dung)
    │
    ▼
DataFrame đã làm sạch + dict thống kê (stats)
    │
    ▼
[4] visualize_cleaning_report()
    Vẽ 4 biểu đồ: bar chart lỗi, pie chart giữ/loại, so sánh trước/sau, scatter word count
```

**Chi tiết từng hàm**:

| Hàm | Tham số | Tác dụng | Trả về |
|---|---|---|---|
| `remove_empty_rows(df, required_columns)` | `df`: DataFrame đầu vào. `required_columns`: danh sách cột bắt buộc (mặc định: `['title', 'main_content', 'url']`) | Thay chuỗi rỗng/khoảng trắng bằng NaN, rồi dùng `dropna(subset=...)` để loại dòng thiếu dữ liệu. | DataFrame đã loại dòng rỗng |
| `validate_urls(df, url_column)` | `df`: DataFrame. `url_column`: tên cột URL (mặc định: `'url'`) | Dùng regex `^https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}` kiểm tra URL. Chỉ giữ dòng có URL hợp lệ. | DataFrame đã loại URL lỗi |
| `filter_by_length(df, content_column, min_words, max_words)` | `content_column`: cột nội dung. `min_words`: tối thiểu (50). `max_words`: tối đa (10.000) | Đếm số từ bằng `str.split()`, giữ bài trong ngưỡng `[min_words, max_words]`. | DataFrame đã lọc |
| `clean_data(df, config)` | `df`: DataFrame thô. `config`: dict tùy chỉnh (từ config.json) | **Hàm tổng hợp**: gọi lần lượt 3 hàm trên, reset index, trả về DataFrame sạch + thống kê chi tiết. | `(DataFrame, dict stats)` |
| `visualize_cleaning_report(stats, df_cleaned, save_dir)` | `stats`: dict thống kê từ `clean_data()`. `df_cleaned`: DataFrame đã sạch (cho scatter plot). `save_dir`: thư mục lưu ảnh | Vẽ 4 biểu đồ PNG: (1) bar chart loại lỗi, (2) pie giữ/loại, (3) so sánh trước/sau, (4) scatter word count với phát hiện outlier IQR. | Không trả về (lưu file ảnh) |

**Thư viện sử dụng trong module này**:
- `re`: Regex kiểm tra URL
- `pandas`: Xử lý DataFrame, dropna, replace
- `matplotlib`: Vẽ biểu đồ
- `seaborn`: Theme và style cho biểu đồ
- `numpy`: Tính IQR cho scatter plot

---

### 5.5 `normalizer.py` — Chuẩn hóa văn bản

**Đường dẫn**: `src/normalizer.py` (441 dòng)

**Mục đích**: Chuẩn hóa văn bản để đưa về dạng đồng nhất, loại bỏ mọi "nhiễu" (HTML tags, emoji, ký tự đặc biệt, URL nhúng, base64) trước khi tiền xử lý NLP. Sau bước này, văn bản chỉ còn chữ cái tiếng Việt (có dấu), chữ số, và khoảng trắng.

**Workflow chi tiết**:

```text
DataFrame đã làm sạch
    │
    ▼
[1] Đếm thống kê nhiễu trước khi xử lý (HTML tags, URL, base64, emoji)
    │
    ▼
[2] Chuẩn hóa 3 cột văn bản chính: title, description, main_content
    Mỗi cột được áp dụng normalize_text() gồm 6 bước nối tiếp:
    │
    │   text.lower()          ← Chuyển toàn bộ về chữ thường
    │       ↓
    │   normalize_unicode()   ← Chuẩn hóa Unicode NFC
    │       ↓
    │   remove_html_tags()    ← Xóa <div>, <p>, <br>, giải mã &amp; &lt;...
    │       ↓
    │   remove_urls_in_text() ← Xóa http://..., data:image/base64,...
    │       ↓
    │   remove_special_characters() ← Xóa emoji, dấu câu, ký tự đặc biệt
    │       ↓
    │   normalize_whitespace() ← Gộp nhiều khoảng trắng thành 1, trim
    │
    ▼
[3] Chuyển 3 cột phụ thành lowercase: tags, category, sub_category
    │
    ▼
[4] visualize_normalize_report(): Vẽ 3 biểu đồ thống kê chuẩn hóa
```

**Chi tiết từng hàm**:

| Hàm | Tác dụng |
|---|---|
| `normalize_unicode(text)` | Dùng `unicodedata.normalize('NFC', text)` để chuẩn hóa Unicode. NFC = Canonical Decomposition → Canonical Composition. Đảm bảo ký tự tiếng Việt có dấu được biểu diễn nhất quán (ví dụ: chữ "ạ" có thể lưu dưới 2 dạng khác nhau, NFC gộp về 1 dạng duy nhất). |
| `remove_html_tags(text)` | (1) `html.unescape(text)`: giải mã HTML entities (`&amp;` → `&`, `&lt;` → `<`). (2) `re.sub(r'<[^>]+>', '', text)`: xóa tất cả thẻ HTML. |
| `remove_urls_in_text(text)` | Xóa URL `https?://\S+` và dữ liệu base64 nhúng `data:image/...;base64,...`. |
| `remove_special_characters(text)` | (1) Xóa emoji bằng regex Unicode range (U+1F600–U+1F64F, v.v.). (2) Chỉ giữ lại: `a-z`, `0-9`, khoảng trắng, và toàn bộ ký tự tiếng Việt có dấu (àáảãạ...ỵđ). Mọi ký tự khác (dấu câu, ký tự đặc biệt) bị thay bằng khoảng trắng. |
| `normalize_whitespace(text)` | Gộp nhiều khoảng trắng liên tiếp thành 1, trim đầu/cuối. |
| `normalize_text(text)` | **Pipeline tổng hợp**: gọi lần lượt 6 bước trên cho 1 đoạn văn bản. |
| `normalize_dataframe(df, text_columns, config)` | **Hàm chính**: áp dụng `normalize_text()` cho 3 cột chính, chuyển 3 cột phụ sang lowercase. Trả về `(DataFrame, dict stats)`. |
| `visualize_normalize_report(stats, save_dir)` | Vẽ 3 biểu đồ: (1) bar chart loại nhiễu, (2) so sánh ký tự trước/sau, (3) pie chart tỷ lệ nhiễu. |

**Tại sao cần chuẩn hóa Unicode NFC?**
Tiếng Việt có dấu có thể được lưu theo 2 cách trong Unicode:
- **NFC (Composed)**: Ký tự `ạ` = 1 mã Unicode duy nhất
- **NFD (Decomposed)**: Ký tự `ạ` = `a` + dấu nặng riêng = 2 mã Unicode

Nếu không chuẩn hóa, 2 chuỗi trông giống nhau nhưng khác mã Unicode → so sánh sai, TF-IDF tính sai.

---

### 5.6 `deduplicator.py` — Loại trùng lặp

**Đường dẫn**: `src/deduplicator.py` (285 dòng)

**Mục đích**: Phát hiện và loại bỏ các bài viết bị lặp lại (cùng nội dung). Do dữ liệu được crawl từ 3 nguồn, nhiều bài có thể bị thu thập trùng.

**Phương pháp**: Sử dụng `pandas.DataFrame.duplicated()` để so sánh trực tiếp giá trị cột `main_content`. Bản ghi xuất hiện đầu tiên được giữ, các bản ghi trùng sau bị loại.

**Workflow chi tiết**:

```text
DataFrame đã chuẩn hóa
    │
    ▼
[1] find_exact_duplicates(df, column='main_content')
    Dùng df.duplicated(subset=[column], keep='first')
    → Tạo mask True/False cho từng dòng (True = trùng)
    │
    ▼
[2] remove_duplicates(df, column, keep)
    Giữ lại dòng có mask = False → reset index
    │
    ▼
[3] generate_dedup_report(df_before, df_after)
    Tính tổng trùng, tỷ lệ %, thống kê theo domain
    In báo cáo trên terminal
    │
    ▼
[4] visualize_dedup_report(df_before, df_after)
    Vẽ 3 biểu đồ: bar domain trước/sau, pie trùng/giữ, so sánh trước/sau
```

**Chi tiết từng hàm**:

| Hàm | Tác dụng |
|---|---|
| `find_exact_duplicates(df, column)` | Dùng `df.duplicated(subset=[column], keep='first')` để tìm bản ghi trùng. Trả về `(DataFrame, mask_series)`. |
| `remove_duplicates(df, column, keep)` | Gọi `find_exact_duplicates()`, loại bỏ dòng trùng (`df[~dup_mask]`), reset index. |
| `generate_dedup_report(df_before, df_after, domain_column)` | Tính thống kê: tổng trước/sau, số đã loại, tỷ lệ %, thống kê theo từng domain. In báo cáo trên terminal. |
| `visualize_dedup_report(df_before, df_after, domain_column, save_dir)` | Vẽ 3 biểu đồ: (1) bar chart domain trước/sau, (2) pie chart giữ/trùng, (3) so sánh tổng trước/sau. |

**Tại sao không dùng MD5 hash?**
Mặc dù README cũ ghi "loại trùng lặp bằng MD5", code thực tế dùng `pandas.duplicated()` (so sánh trực tiếp giá trị cột). Phương pháp này đơn giản hơn và đủ chính xác vì: (1) dữ liệu đã được chuẩn hóa ở bước trước nên các bài giống nhau sẽ có giá trị cột giống hệt; (2) `duplicated()` của pandas được tối ưu cho tốc độ.

---

### 5.7 `preprocessing.py` — Tiền xử lý NLP & TF-IDF

**Đường dẫn**: `src/preprocessing.py` (407 dòng)

**Mục đích**: Biến nội dung bài viết (văn bản tự nhiên) thành vector số (ma trận TF-IDF) để máy học có thể xử lý. Đây là bước cầu nối giữa dữ liệu văn bản và các thuật toán phân cụm/phân loại.

**Workflow chi tiết**:

```text
cleaned_news.csv (DataFrame đã làm sạch)
    │
    ▼
[1] Tạo TextPreprocessor
    - Nạp danh sách 113 từ dừng tiếng Việt
    - Khởi tạo TfidfVectorizer(max_features=3000, min_df=5, max_df=0.7)
    │
    ▼
[2] process_text() cho từng bài viết (có progress bar)
    - text.lower()                    ← Chuyển chữ thường
    - word_tokenize(text, format="text")  ← Tách từ tiếng Việt bằng underthesea
    - Loại từ dừng + từ chứa số + ký tự đơn lẻ
    → Tạo cột processed_content
    │
    ▼
[3] fit_transform_tfidf(corpus)
    TfidfVectorizer.fit_transform() → Ma trận TF-IDF sparse (n_docs × 3000)
    │
    ▼
[4] visualize_preprocessing_report()
    Vẽ 3 biểu đồ TRƯỚC KHI giảm chiều (để giữ feature names)
    │
    ▼
[5] reduce_dimension(X, n_components=300)  [nếu có cấu hình]
    TruncatedSVD → Ma trận dense (n_docs × 300)
    │
    ▼
[6] save_features()
    Lưu ma trận TF-IDF → data/tfidf_features.pkl
    Lưu vectorizer → models/vectorizer.pkl
    │
    ▼
Output: (DataFrame có cột processed_content, Ma trận đặc trưng)
```

**Class `TextPreprocessor`**:

| Phương thức | Tác dụng |
|---|---|
| `__init__(stopword_path, max_features)` | Khởi tạo: nạp stopwords, tạo `TfidfVectorizer(max_features, min_df=5, max_df=0.7)`. `min_df=5` = từ phải xuất hiện trong ít nhất 5 tài liệu; `max_df=0.7` = từ xuất hiện trong >70% tài liệu bị bỏ qua. |
| `_load_stopwords()` | Đọc file `vietnamese-stopwords.txt`, mỗi dòng là 1 từ dừng. Thêm cả dạng có gạch dưới (ví dụ: "có thể" → thêm "có_thể"). Bỏ qua dòng trống và dòng bắt đầu bằng `#`. |
| `process_text(text)` | Xử lý 1 đoạn văn bản: (1) lowercase, (2) tách từ tiếng Việt bằng `underthesea.word_tokenize()`, (3) loại từ dừng + từ chứa số + ký tự đơn lẻ. |
| `fit_transform_tfidf(corpus)` | Gọi `self.vectorizer.fit_transform(corpus)`. Biến tập văn bản thành ma trận TF-IDF sparse. |
| `reduce_dimension(X, n_components)` | Dùng `TruncatedSVD` (= Latent Semantic Analysis) giảm chiều từ 3000 xuống 300. Giúp giảm nhiễu và tăng tốc các bước sau. |
| `save_features(X, matrix_path, vectorizer_path)` | Dùng `joblib.dump()` lưu ma trận và vectorizer ra file `.pkl`. |

**Hàm độc lập**:

| Hàm | Tác dụng |
|---|---|
| `encode_labels(df, label_column, encoding_method)` | Mã hóa cột nhãn. Nếu `method='label'`: dùng `LabelEncoder` (mỗi nhãn → 1 số). Nếu `method='onehot'`: dùng `OneHotEncoder`. |
| `encode_source(df, source_column)` | Mã hóa cột nguồn bằng `OneHotEncoder`. Mỗi giá trị nguồn tạo 1 cột mới (0/1). |
| `preprocess_nlp(df, ...)` | **Hàm chính**: tạo `TextPreprocessor`, xử lý văn bản, tạo TF-IDF, vẽ biểu đồ, giảm chiều (nếu có), lưu output. Trả về `(DataFrame, ma trận đặc trưng)`. |
| `visualize_preprocessing_report(df, X, vectorizer, ...)` | Vẽ 3 biểu đồ: (1) histogram so sánh số từ trước/sau, (2) bar chart trung bình từ trước/sau/đã loại, (3) top 20 từ/cụm từ nổi bật theo điểm TF-IDF. |

**Giải thích TF-IDF**:
- **TF (Term Frequency)**: Tần suất từ xuất hiện trong 1 tài liệu. Từ nào xuất hiện nhiều trong bài → TF cao.
- **IDF (Inverse Document Frequency)**: Nghịch đảo tần suất tài liệu. Từ nào xuất hiện trong ít tài liệu → IDF cao (từ đó có tính phân biệt cao).
- **TF-IDF = TF × IDF**: Kết hợp cả 2 → từ nào quan trọng trong 1 bài nhưng hiếm trong toàn tập → điểm cao.

**Giải thích TruncatedSVD (Latent Semantic Analysis)**:
- Ma trận TF-IDF gốc có 3000 cột (3000 từ), nhiều cột có tương quan với nhau (ví dụ: "AI" và "trí tuệ nhân tạo").
- TruncatedSVD giảm từ 3000 → 300 chiều, giữ lại thông tin quan trọng nhất, bỏ nhiễu.
- Giúp KMeans và MLP chạy nhanh hơn và chính xác hơn.

---

### 5.8 `modeling.py` — Phân cụm & Gán nhãn

**Đường dẫn**: `src/modeling.py` (1107 dòng)

**Mục đích**: Phân cụm tự động các bài viết thành 5 nhóm chủ đề bằng KMeans, gán nhãn cho từng cụm, so sánh với DBSCAN, và chia dữ liệu thành train/val/test.

**Workflow chi tiết**:

```text
data/tfidf_features.pkl + models/vectorizer.pkl + data/processed_news.csv
    │
    ▼
[1] load_inputs()
    Nạp ma trận TF-IDF, vectorizer, DataFrame đã xử lý
    │
    ▼
[2] plot_elbow(X, k_range=2..10)
    Chạy KMeans với K=2,3,...,10 → vẽ đồ thị Elbow
    Để tìm K tối ưu (điểm "khuỷu tay" nơi inertia giảm chậm lại)
    │
    ▼
[3] cluster_and_label(df, X, vectorizer)
    ├── run_kmeans(X, k=5) → Chạy KMeans với K=5
    ├── evaluate_clusters() → Tính Silhouette Score, Davies-Bouldin Score
    ├── get_top_keywords() → Lấy top 15 từ khóa quan trọng nhất mỗi cụm
    │   (Tính lại TF-IDF cho từng cụm, KHÔNG dùng cluster_centers_
    │    vì centers nằm trong không gian SVD đã giảm chiều)
    └── Gán nhãn theo CLUSTER_MAPPING:
        Cluster 0 → "Mobile"
        Cluster 1 → "Tri_tue_nhan_tao"
        Cluster 2 → "Chuyen_doi_so"
        Cluster 3 → "Mobile"
        Cluster 4 → "Hang_khong_vu_tru"
    │
    ▼
[4] run_dbscan(X, eps=0.7, min_samples=5, metric='cosine')
    Chạy DBSCAN để so sánh kết quả với KMeans
    Lưu ý: DBSCAN dùng brute-force O(n²) với metric='cosine'
    │
    ▼
[5] export_sample_for_labeling(df, n_per_cluster=200)
    Trích 200 bài/cụm × 5 cụm = 1000 bài để gán nhãn tay (kiểm chứng)
    │
    ▼
[6] split_dataset(df, test_size=0.30)
    Chia dữ liệu: 70% train, 15% validation, 15% test
    Stratified split: giữ tỷ lệ các lớp cân bằng giữa 3 tập
    │
    ▼
[7] Lưu file + vẽ biểu đồ
    - labeled_news.csv, clustered_news.csv
    - train.csv, val.csv, test.csv
    - Biểu đồ: cluster_distribution, elbow, PCA, silhouette, heatmap, wordcloud...
```

**Cấu hình mặc định `DEFAULT_CONFIG`**:

| Khóa | Giá trị | Ý nghĩa |
|---|---|---|
| `n_clusters` | `5` | Số cụm KMeans |
| `k_search_range` | `range(2, 11)` | Dải K thử cho Elbow Method |
| `dbscan_eps` | `0.7` | Bán kính epsilon cho DBSCAN |
| `dbscan_min_samples` | `5` | Số điểm tối thiểu trong 1 cụm DBSCAN |
| `dbscan_metric` | `"cosine"` | Metric khoảng cách cho DBSCAN |
| `random_state` | `42` | Seed cho kết quả tái lập được |

**Chi tiết các hàm quan trọng**:

| Hàm | Tác dụng |
|---|---|
| `load_inputs(tfidf_path, vec_path, data_path)` | Nạp 3 file đầu vào: ma trận TF-IDF (joblib), vectorizer (joblib), DataFrame (CSV). |
| `plot_elbow(X, k_range, output_path)` | Chạy KMeans với nhiều giá trị K, vẽ đồ thị inertia vs K để tìm K tối ưu. |
| `run_kmeans(X, k, random_state)` | Chạy `KMeans(n_clusters=k, n_init=10)`. `n_init=10`: chạy 10 lần với centroid khác nhau, lấy kết quả tốt nhất. |
| `evaluate_clusters(X, clusters)` | Tính 2 chỉ số: **Silhouette Score** (càng gần 1 càng tốt, cho biết các điểm trong cụm gần nhau và xa cụm khác) và **Davies-Bouldin Score** (càng nhỏ càng tốt, cho biết mức độ tách biệt giữa các cụm). |
| `get_top_keywords(df, vectorizer, clusters, n_clusters)` | **Quan trọng**: Lấy top từ khóa bằng cách tính lại TF-IDF GỐC (không dùng cluster_centers_ vì nó nằm trong không gian SVD đã giảm chiều, không tương ứng 1-1 với tên từ). |
| `cluster_and_label(df, X, vectorizer, config, cluster_mapping)` | **Hàm chính**: chạy KMeans, đánh giá, lấy keywords, gán nhãn theo mapping. Trả về `(DataFrame_có_cột_cluster_và_label, kmeans_model, keywords)`. |
| `run_dbscan(X, eps, min_samples, metric, sample_size)` | Chạy DBSCAN. Có tùy chọn `sample_size` để lấy mẫu ngẫu nhiên nếu dữ liệu quá lớn (tránh tràn RAM do O(n²)). |
| `split_dataset(df, label_column, test_size)` | Chia stratified: 70/15/15. Dùng `train_test_split()` 2 lần: lần 1 chia 70/30, lần 2 chia 30 thành 15/15. |
| `export_sample_for_labeling(df, output_path, n_per_cluster)` | Trích 200 bài/cụm để gán nhãn tay kiểm chứng. Thêm cột `manual_label` trống. |
| `compare_manual_vs_kmeans(labeled_path)` | So sánh nhãn tay với nhãn KMeans → tính % chính xác. |
| `safe_to_csv(df, path)` | Lưu CSV với xử lý lỗi PermissionError rõ ràng (lỗi hay gặp trên Windows khi file đang mở). |

**Các hàm vẽ biểu đồ**:

| Hàm | Biểu đồ |
|---|---|
| `plot_cluster_distribution()` | Phân bố bài viết theo cụm (bar chart) |
| `plot_kmeans_vs_dbscan()` | So sánh số cụm KMeans vs DBSCAN |
| `plot_pca()` | PCA 2D visualization — chiếu dữ liệu xuống 2 chiều để xem các cụm có tách biệt không |
| `plot_top_tfidf()` | Top 20 từ/cụm từ nổi bật theo TF-IDF |
| `save_tfidf_matrix()` | Lưu bảng ma trận TF-IDF (30 dòng × 20 cột) ra CSV |
| `plot_tfidf_heatmap()` | Heatmap ma trận TF-IDF (50 dòng × 30 cột) |
| `plot_silhouette()` | Silhouette analysis — hình quạt cho từng cụm |
| `plot_cluster_distance()` | Khoảng cách trung bình từ mỗi điểm đến centroid gần nhất |
| `plot_cluster_centers_heatmap()` | Heatmap giá trị centroid cho top 20 từ quan trọng |
| `plot_dataset_split_pie()` | Pie chart tỷ lệ train/val/test (70/15/15) |
| `generate_wordclouds()` | Word cloud cho từng cụm (sử dụng thư viện `wordcloud`) |
| `plot_dbscan_distribution()` | Phân bố cụm DBSCAN |

---

### 5.9 `deep_learning_model.py` — Huấn luyện mô hình Deep Learning

**Đường dẫn**: `src/deep_learning_model.py` (495 dòng)

**Mục đích**: Huấn luyện mô hình phân loại văn bản từ dữ liệu đã gán nhãn (output của bước phân cụm). Hỗ trợ 2 backend: `sklearn_mlp` (mặc định, không cần TensorFlow) và `tensorflow_dense` (cần cài TensorFlow).

**Workflow chi tiết**:

```text
data/train.csv + data/val.csv + data/test.csv
(Nếu chưa có → fallback về data/labeled_news.csv → tự chia 70/15/15)
    │
    ▼
[1] _load_training_frames()
    Đọc 3 file CSV. Nếu thiếu → đọc labeled_news.csv và tự chia.
    Nếu không tìm thấy file nào → trả status "skipped"
    │
    ▼
[2] _pick_text_column()
    Chọn cột văn bản: ưu tiên 'processed_content', fallback 'main_content'
    │
    ▼
[3] _load_vectorizer()
    Nạp TF-IDF vectorizer đã fit từ models/vectorizer.pkl
    │
    ▼
[4] vectorizer.transform()
    Biến văn bản train/val/test thành vector TF-IDF
    (Dùng transform, KHÔNG fit lại — vectorizer đã fit ở bước NLP)
    │
    ▼
[5] LabelEncoder.fit_transform()
    Mã hóa nhãn text ("Mobile", "AI"...) → số (0, 1, 2...)
    │
    ▼
[6] Train model
    ├── Backend sklearn_mlp: MLPClassifier(hidden_layer_sizes=(128,64),
    │   learning_rate_init=0.001, max_iter=30, early_stopping=True)
    │
    └── Backend tensorflow_dense: Sequential(Dense(128,relu) → Dropout(0.3)
        → Dense(64,relu) → Dropout(0.2) → Dense(n_classes,softmax))
    │
    ▼
[7] Đánh giá trên val và test
    - Accuracy, F1-score (macro)
    - Classification report (precision/recall/f1 cho từng lớp)
    - Confusion matrix
    │
    ▼
[8] Lưu output
    - models/deep_learning_text_model.pkl (model + label_encoder + metadata)
    - models/deep_learning_metrics.json (các chỉ số đánh giá)
    - data/deep_learning_confusion_matrix.csv
    - 4 biểu đồ PNG trong notebooks/
```

**Cấu hình `DEFAULT_TRAINING_CONFIG`**:

| Khóa | Giá trị | Ý nghĩa |
|---|---|---|
| `backend` | `"sklearn_mlp"` | Loại mô hình: `sklearn_mlp` (MLPClassifier) hoặc `tensorflow_dense` (Keras Dense) |
| `hidden_layer_sizes` | `[128, 64]` | Kiến trúc mạng nơ-ron: 2 lớp ẩn (128 và 64 nơ-ron) |
| `learning_rate_init` | `0.001` | Learning rate ban đầu |
| `batch_size` | `64` | Số mẫu trong mỗi batch |
| `max_iter` | `30` | Số epoch tối đa |
| `random_state` | `42` | Seed cho kết quả tái lập |

**Chi tiết các hàm**:

| Hàm | Tác dụng |
|---|---|
| `_load_training_frames(project_root, config)` | Đọc train/val/test CSV. Nếu thiếu, fallback về labeled_news.csv hoặc clustered_news.csv, rồi tự chia. Trả về 3 DataFrame hoặc `(None, None, None)`. |
| `_pick_text_column(df, config)` | Chọn cột văn bản: ưu tiên `processed_content`, fallback `main_content`. |
| `_train_sklearn_mlp(X_train, y_train, config)` | Tạo và train `MLPClassifier` với early_stopping=True (dừng sớm nếu validation loss không giảm). |
| `_train_tensorflow_dense(X_train, y_train, X_val, y_val, config)` | Tạo và train mô hình Keras Sequential. Gồm: Input → Dense(128,relu) → Dropout(0.3) → Dense(64,relu) → Dropout(0.2) → Dense(n_classes,softmax). Loss: sparse_categorical_crossentropy. |
| `_predict(model, X, backend)` | Dự đoán. Nếu TensorFlow: argmax trên output softmax. Nếu sklearn: dùng model.predict(). |
| `train_deep_learning_model(project_root, config)` | **Hàm chính**: orchestrate toàn bộ workflow train. Trả về dict metrics. Nếu chưa có dữ liệu → trả `{"status": "skipped"}`. |
| `plot_existing_results(project_root, config)` | Vẽ lại biểu đồ từ metrics JSON và confusion matrix CSV có sẵn (không cần train lại). |
| `save_confusion_matrix_plot(cm, labels, path)` | Vẽ heatmap confusion matrix với số trong từng ô. |
| `save_metrics_plot(metrics, path)` | Vẽ bar chart 4 giá trị: Val Accuracy, Val F1, Test Accuracy, Test F1. |
| `save_class_f1_plot(metrics, path)` | Vẽ horizontal bar F1-score cho từng lớp. |
| `save_dataset_split_plot(metrics, path)` | Vẽ bar chart số lượng bài trong train/val/test. |

**Giải thích MLPClassifier (Multi-Layer Perceptron)**:
- Mạng nơ-ron feedforward (truyền thẳng) với 2 lớp ẩn.
- Input: vector TF-IDF (3000 chiều hoặc sau SVD).
- Hidden 1: 128 nơ-ron, activation ReLU.
- Hidden 2: 64 nơ-ron, activation ReLU.
- Output: n lớp (softmax tự động trong sklearn).
- Early stopping: dùng 10% dữ liệu train làm validation, dừng nếu không cải thiện.

---

### 5.10 `predict.py` — Dự đoán dữ liệu mới

**Đường dẫn**: `src/predict.py` (75 dòng)

**Mục đích**: Nạp mô hình đã train và dự đoán nhãn chủ đề cho văn bản mới. Đây là file sử dụng sau khi toàn bộ pipeline đã chạy xong.

**Workflow**:

```text
Văn bản mới (đã tiền xử lý giống processed_content)
    │
    ▼
[1] _load_model_bundle()
    Nạp models/deep_learning_text_model.pkl
    (chứa: model, label_encoder, backend, text_column)
    │
    ▼
[2] _load_vectorizer()
    Nạp models/vectorizer.pkl
    │
    ▼
[3] vectorizer.transform(texts)
    Biến văn bản mới thành vector TF-IDF
    │
    ▼
[4] model.predict(X)
    Dự đoán nhãn số
    │
    ▼
[5] label_encoder.inverse_transform(pred_ids)
    Chuyển nhãn số → nhãn text ("Mobile", "AI"...)
    │
    ▼
Output: list nhãn dự đoán
```

**Chi tiết từng hàm**:

| Hàm | Tác dụng |
|---|---|
| `_project_root()` | Tính đường dẫn thư mục gốc dự án từ vị trí file hiện tại. |
| `_load_model_bundle(project_root, model_path)` | Nạp model bundle đã train. Kiểm tra file tồn tại, raise lỗi nếu thiếu. |
| `_load_vectorizer(project_root, vectorizer_path)` | Nạp TF-IDF vectorizer đã fit. |
| `predict_texts(texts, project_root, model_path, vectorizer_path)` | **Hàm chính**: nhận danh sách text (hoặc 1 text), nạp model + vectorizer, transform, predict, chuyển nhãn. Hỗ trợ cả backend sklearn_mlp và tensorflow_dense. |

**Lưu ý quan trọng**: Văn bản đầu vào cho `predict_texts()` phải được tiền xử lý **giống hệt** cách tạo `processed_content` (lowercase, tách từ tiếng Việt, loại stopwords). Nếu không, kết quả dự đoán sẽ sai.

---

### 5.11 `pipeline.py` — Điều phối Pipeline

**Đường dẫn**: `src/pipeline.py` (818 dòng)

**Mục đích**: File trung tâm điều phối toàn bộ pipeline. Import và gọi các module khác theo thứ tự. Hỗ trợ chạy toàn bộ hoặc chạy riêng từng phần theo thành viên.

**Class `DataPipeline`**:

```text
DataPipeline
├── __init__(config_path)       → Nạp config, khởi tạo biến
├── _load_config()              → Đọc config.json
├── _default_config()           → Cấu hình mặc định nếu thiếu file
├── _get_path(relative_path)    → Chuyển đường dẫn tương đối → tuyệt đối
├── _save_checkpoint(step_name) → Lưu checkpoint CSV sau mỗi bước
│
├── step_integration()          → Bước 1: Gọi integration.integrate_data()
├── step_load_data()            → Bước 1.5: Đọc CSV thô
├── step_clean()                → Bước 2: Gọi cleaning.clean_data()
├── step_normalize()            → Bước 3: Gọi normalizer.normalize_dataframe()
├── step_deduplicate()          → Bước 4: Gọi deduplicator.remove_duplicates()
├── step_preprocess_nlp()       → Bước 5: Gọi preprocessing.preprocess_nlp()
├── step_encode_labels()        → Bước 6: Gọi preprocessing.encode_labels()
├── step_encode_source()        → Bước 7: Gọi preprocessing.encode_source()
├── step_clustering()           → Bước 8: PLACEHOLDER (chờ tích hợp)
├── step_train_model()          → Bước 9: Gọi deep_learning_model.train_deep_learning_model()
├── step_export()               → Bước 10: PLACEHOLDER
│
├── save_output(output_config_key) → Lưu DataFrame ra CSV
├── _visualize_pipeline_funnel()   → Vẽ biểu đồ waterfall pipeline
│
├── run_all()                   → Chạy toàn bộ pipeline
├── run_step(step_name)         → Chạy 1 bước cụ thể
│
├── run_tv1()  → Phần TV1: Thu thập & EDA
├── run_tv2()  → Phần TV2: Tích hợp, Làm sạch, Chuẩn hóa, Loại trùng
├── run_tv3()  → Phần TV3: Tiền xử lý NLP & TF-IDF
├── run_tv4()  → Phần TV4: Phân cụm & Gán nhãn
├── run_tv5()  → Phần TV5: Deep Learning & Tuning
└── run_tv6()  → Phần TV6: Xuất dữ liệu & Báo cáo
```

**Cách chạy** (2 mode):

```bash
# Mode 1: Argument dòng lệnh
python src/pipeline.py all    # Chạy toàn bộ
python src/pipeline.py tv2    # Chạy phần TV2

# Mode 2: Menu tương tác (không có argument)
python src/pipeline.py
# → Hiện menu chọn 0-6 hoặc q để thoát
```

**Method chính `run_all()`** thực hiện:

```text
1. step_integration()         ← Tích hợp dữ liệu từ 3 nguồn
2. step_clean()               ← Làm sạch
3. step_normalize()           ← Chuẩn hóa
4. step_deduplicate()         ← Loại trùng
5. save_output('output_cleaned_path')  ← Lưu cleaned_news.csv
6. step_preprocess_nlp()      ← NLP + TF-IDF
7. save_output('output_processed_path') ← Lưu processed_news.csv
8. _visualize_pipeline_funnel() ← Vẽ biểu đồ waterfall
9. step_encode_labels()       ← Mã hóa nhãn
10. step_encode_source()      ← Mã hóa nguồn
11. log_summary()             ← In tổng kết
```

**Cơ chế checkpoint**: Sau mỗi bước, `_save_checkpoint()` lưu DataFrame hiện tại vào `data/checkpoints/{step_name}.csv`. Giúp debug và chạy lại từ giữa nếu pipeline bị lỗi.

**Cơ chế method chaining**: Mỗi method `step_*()` trả về `self`, cho phép viết dạng chain:
```python
pipeline.step_load_data().step_clean().step_normalize()
```

---

### 5.12 `visualize_collection.py` — Trực quan hóa EDA

**Đường dẫn**: `src/visualize_collection.py` (390 dòng)

**Mục đích**: Vẽ 8 biểu đồ trực quan hóa cho phần Data Collection & EDA (Exploratory Data Analysis). Đọc dữ liệu thô `news_data.csv` và dữ liệu đã phân cụm `clustered_news.csv`.

**Workflow**:

```text
news_data.csv + clustered_news.csv
    │
    ▼
run_collection_viz()
    ├── _fig1_metric_cards()          → 4 thẻ metric tổng quan
    ├── _fig2_domain_donut()          → Donut phân bổ nguồn báo
    ├── _fig3_subcategory()           → Bar chart sub-category
    ├── _fig4_timeline()              → Line chart timeline theo tháng
    ├── _fig5_cluster_donut()         → Donut nhãn K-Means
    ├── _fig6_top_tags()              → Bar chart top tags
    ├── _fig7_content_length()        → Histogram độ dài nội dung
    └── _fig8_dashboard()             → Dashboard tổng hợp 2×3
```

**Chi tiết 8 biểu đồ**:

| Biểu đồ | Hàm | Nội dung |
|---|---|---|
| Hình 1 | `_fig1_metric_cards()` | 4 thẻ tổng quan: Tổng bài viết, Số nguồn báo (3), Số nhãn K-Means, Avg content length |
| Hình 2 | `_fig2_domain_donut()` | Donut chart: VnExpress / VietnamNet / Thanh Niên — % và số lượng |
| Hình 3 | `_fig3_subcategory()` | Horizontal bar chart: top 8 danh mục con |
| Hình 4 | `_fig4_timeline()` | Line chart + area fill: số bài theo tháng (2024–2026), đánh dấu Jan 2025 |
| Hình 5 | `_fig5_cluster_donut()` | Donut chart: phân bổ 5 nhãn K-Means |
| Hình 6 | `_fig6_top_tags()` | Horizontal bar chart: top 9 tags nổi bật |
| Hình 7 | `_fig7_content_length()` | Bar chart: histogram phân bổ độ dài nội dung (0–500, 500–1k, ..., 7k+) |
| Hình 8 | `_fig8_dashboard()` | Dashboard tổng hợp 2×3: gộp 5 biểu đồ nhỏ vào 1 hình |

---

### 5.13 `visualize_extra.py` — Biểu đồ bổ sung

**Đường dẫn**: `src/visualize_extra.py` (222 dòng)

**Mục đích**: Vẽ 2 biểu đồ bổ sung cho phần Data Collection.

**Workflow**:

```text
news_data.csv
    │
    ▼
run_extra_viz()
    ├── _fig_heatmap()   → Correlation Matrix: Nguồn báo vs Top Tags
    └── _fig_scatter()   → Scatter Plot: Word Count từng bài + outliers
```

**Chi tiết**:

| Biểu đồ | Hàm | Nội dung |
|---|---|---|
| Hình 9 | `_fig_heatmap()` | Heatmap đỏ–trắng: ma trận số bài theo (nguồn báo × tag). Dùng `LinearSegmentedColormap` custom. |
| Hình 10 | `_fig_scatter()` | Scatter plot: mỗi bài là 1 điểm, trục Y = word count. Điểm xanh = normal, tam giác đỏ = outlier (IQR). Vùng normal được shade xanh nhạt. |

---

## 6. Luồng dữ liệu giữa các file

Bảng dưới đây cho biết mỗi file tạo ra output gì và file nào sử dụng output đó:

| File tạo ra | Output | File sử dụng |
|---|---|---|
| `integration.py` | `data/news_data.csv` | `pipeline.py` / `cleaning.py` |
| `cleaning.py` | DataFrame đã làm sạch + stats dict | `normalizer.py` (qua pipeline) |
| `normalizer.py` | DataFrame đã chuẩn hóa + stats dict | `deduplicator.py` (qua pipeline) |
| `deduplicator.py` | DataFrame đã loại trùng | `pipeline.py` → lưu `cleaned_news.csv` |
| `pipeline.py` (lưu) | `data/cleaned_news.csv` | `preprocessing.py` |
| `preprocessing.py` | `data/processed_news.csv` + `data/tfidf_features.pkl` + `models/vectorizer.pkl` | `modeling.py`, `deep_learning_model.py`, `predict.py` |
| `modeling.py` | `data/labeled_news.csv` + `data/train.csv` + `data/val.csv` + `data/test.csv` + `models/kmeans_model.pkl` | `deep_learning_model.py` |
| `deep_learning_model.py` | `models/deep_learning_text_model.pkl` + `models/deep_learning_metrics.json` | `predict.py` |
| `predict.py` | Nhãn dự đoán (list) | Ứng dụng demo |

---

## 7. Phân công Thành viên

| Thành viên | Tên | Phụ trách | Module chính | Lệnh chạy riêng |
|---|---|---|---|---|
| TV1 | Ngô Hoàng Anh | Thu thập dữ liệu & EDA | `visualize_collection.py`, `visualize_extra.py` | `python src/pipeline.py tv1` |
| TV2 | Lê Quang Thi | Tích hợp, Làm sạch, Chuẩn hóa, Loại trùng | `integration.py`, `cleaning.py`, `normalizer.py`, `deduplicator.py` | `python src/pipeline.py tv2` |
| TV3 | Tôn Hoàng Nhớ | Tiền xử lý NLP & TF-IDF | `preprocessing.py` | `python src/pipeline.py tv3` |
| TV4 | Nguyễn Văn Trường | Phân cụm & Gán nhãn | `modeling.py` | `python src/pipeline.py tv4` |
| TV5 | (Thành viên 5) | Deep Learning & Tuning | `deep_learning_model.py` | `python src/pipeline.py tv5` |
| TV6 | Khánh Huyền | Xuất dữ liệu & Báo cáo | (placeholder) | `python src/pipeline.py tv6` |

**Phụ thuộc giữa các phần**:

```text
TV1 (Thu thập)
    ↓  vnexpress.csv, thanhnien.csv, vietnamnet.csv
TV2 (Tích hợp & Làm sạch)
    ↓  cleaned_news.csv
TV3 (NLP)
    ↓  processed_news.csv + tfidf_features.pkl + vectorizer.pkl
TV4 (Phân cụm)
    ↓  labeled_news.csv + train/val/test.csv + kmeans_model.pkl
TV5 (Deep Learning)
    ↓  deep_learning_text_model.pkl
TV6 (Báo cáo)
```

Mỗi thành viên **phải chạy các bước trước** của mình hoặc có sẵn output của bước trước. Ví dụ: TV3 cần `cleaned_news.csv` (output TV2) trước khi chạy.

---

## 8. Cách chạy

### 8.1 Cài đặt

```bash
pip install -r requirements.txt
```

### 8.2 Chạy toàn bộ pipeline

```bash
python src/pipeline.py all
```

### 8.3 Chạy riêng từng phần

```bash
python src/pipeline.py tv1    # Thu thập & EDA
python src/pipeline.py tv2    # Làm sạch, Chuẩn hóa, Loại trùng
python src/pipeline.py tv3    # Tiền xử lý NLP & TF-IDF
python src/pipeline.py tv4    # Phân cụm & Gán nhãn
python src/pipeline.py tv5    # Deep Learning & Tuning
python src/pipeline.py tv6    # Xuất dữ liệu & Báo cáo
```

### 8.4 Chạy module độc lập

```bash
python src/modeling.py                  # Chạy riêng phân cụm
python src/deep_learning_model.py       # Chạy riêng huấn luyện DL
python src/visualize_collection.py      # Chạy riêng vẽ biểu đồ EDA
python src/visualize_extra.py           # Chạy riêng biểu đồ bổ sung
```

### 8.5 Dự đoán dữ liệu mới

```python
from predict import predict_texts

texts = [
    "tri_tue_nhan_tao hoc_may cong_nghe so",
    "iphone samsung man_hinh camera pin",
]
labels = predict_texts(texts)
# Output: ["Tri_tue_nhan_tao", "Mobile"]
```

### 8.6 Cấu hình

Sửa file `config.json` để thay đổi:
- Đường dẫn input/output
- Ngưỡng lọc độ dài (min_words, max_words)
- Số chiều TF-IDF (max_features, n_components)
- Phương pháp encoding

---

> **Ghi chú cho agent**: File này mô tả toàn bộ codebase ở thời điểm viết. Khi cần hiểu code, đọc file này trước rồi mới đọc source code cụ thể. Các hàm, class, và workflow đã được ghi chú đầy đủ. Nếu cần sửa code, tham khảo phần workflow và luồng dữ liệu để hiểu impact của thay đổi.

> **Ghi chú cho sinh viên**: Khi phản biện, tập trung vào: (1) Tại sao chọn phương pháp này? (2) Kết quả ra sao? (3) Có gì có thể cải thiện? Mỗi module đã có docstring giải thích "tại sao" bên cạnh "làm gì". Các con số (min_words=50, max_features=3000, n_clusters=5, hidden_layer_sizes=[128,64]) đều có thể được hỏi lý do.

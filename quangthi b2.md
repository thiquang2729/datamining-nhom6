# Kế hoạch Thực hiện - Thành viên 2: Lê Quang Thi

## Vai trò: Data Quality & Pipeline Engineer (Làm sạch & Tự động hóa)

---

## Bối cảnh

Dự án **Pipeline Tinh chế và Phân loại Tin tức Công nghệ** nhằm xây dựng hệ thống xử lý dữ liệu từ tin tức thô → dữ liệu sạch có nhãn phân loại (5 thể loại: Phần cứng, Phần mềm, AI, Thiết bị di động, An ninh mạng).

**Dữ liệu đầu vào hiện có:**
- File: [news_data.csv](file:///d:/Dev/Code/Nhom8_DataMining_NEWS/data/news_data.csv) (~90MB, ~19.120 dòng)
- Các cột: `article_id`, `url`, `domain`, `category_url`, `author_url`, `amp_url`, `reference_url`, `short_url`, `title`, `description`, `main_content`, `author_name`, `tags`, `thumbnail_url`, `images`, `videos`, `published_time`, `scraped_time`, `category`, `sub_category`, `views_count`, `comments_count`
- Dữ liệu có vấn đề encoding (Unicode bị lỗi khi đọc bằng PowerShell), cần kiểm tra lại encoding thực tế

---

## Tổng quan Nhiệm vụ

Thành viên 2 chịu trách nhiệm **4 nhiệm vụ chính**:

| # | Nhiệm vụ | File tạo mới | Mức độ |
|---|----------|-------------|--------|
| 1 | Làm sạch dữ liệu rỗng/lỗi | `src/cleaning.py` | ⭐⭐ |
| 2 | Chuẩn hóa Unicode & loại bỏ HTML/ký tự đặc biệt | `src/normalizer.py` | ⭐⭐⭐ |
| 3 | Xử lý trùng lặp (kiểm tra bài viết lặp lại) | `src/deduplicator.py` | ⭐⭐ |
| 4 | Lắp ráp Pipeline hoàn chỉnh | `src/pipeline.py` | ⭐⭐⭐⭐ |

---

## Giai đoạn 1: Làm sạch dữ liệu rỗng/lỗi

#### [NEW] [cleaning.py](file:///d:/Dev/Code/Nhom8_DataMining_NEWS/src/cleaning.py)

**Mục tiêu:** Loại bỏ các bản ghi không hợp lệ khỏi tập dữ liệu thô.

**Logic xử lý chi tiết:**

1. **Loại bỏ dòng rỗng (Missing Data):**
   - Kiểm tra các cột quan trọng: `title`, `main_content`, `url`
   - Xóa bản ghi nếu `title` HOẶC `main_content` là `NaN`/rỗng
   - Log số lượng bản ghi bị xóa

2. **Loại bỏ URL lỗi:**
   - Validate URL bằng regex: phải bắt đầu bằng `http://` hoặc `https://`
   - Kiểm tra URL có chứa domain hợp lệ không
   - Xóa bản ghi có URL không hợp lệ

3. **Lọc bài viết quá ngắn/quá dài:**
   - Đếm số từ trong `main_content`
   - Loại bỏ bài viết có **< 50 từ** (quá ngắn, không đủ thông tin)
   - Loại bỏ bài viết có **> 10.000 từ** (có thể là dữ liệu lỗi, cào nhầm nhiều trang)
   - Ngưỡng có thể cấu hình qua tham số

**Công cụ sử dụng:** `pandas`, `re` (regex)

**Đầu vào:** `data/news_data.csv`  
**Đầu ra:** DataFrame đã loại bỏ các bản ghi lỗi + báo cáo thống kê

```python
# Cấu trúc hàm chính
def remove_empty_rows(df, required_columns=['title', 'main_content', 'url']):
    """Xóa các dòng có giá trị rỗng ở các cột bắt buộc"""

def validate_urls(df, url_column='url'):
    """Kiểm tra và loại bỏ các URL không hợp lệ"""

def filter_by_length(df, content_column='main_content', min_words=50, max_words=10000):
    """Lọc bài viết theo độ dài nội dung"""

def clean_data(df):
    """Hàm tổng hợp chạy toàn bộ quy trình làm sạch"""
```

**Trực quan hóa kết quả Cleaning (3 biểu đồ):**

| # | File | Mô tả |
|---|------|-------|
| 1 | `cleaning_01_error_types.png` | Bar chart chi tiết từng loại lỗi đã loại bỏ (dòng rỗng, quá ngắn, quá dài) |
| 2 | `cleaning_02_pie_chart.png` | Pie chart tỷ lệ giữ lại vs loại bỏ |
| 3 | `cleaning_03_comparison.png` | Bar chart so sánh số bản ghi trước/sau làm sạch |

---

## Giai đoạn 2: Chuẩn hóa Unicode & Loại bỏ HTML/Ký tự đặc biệt

#### [NEW] [normalizer.py](file:///d:/Dev/Code/Nhom8_DataMining_NEWS/src/normalizer.py)

**Mục tiêu:** Chuẩn hóa toàn bộ văn bản về dạng thống nhất, loại bỏ nhiễu.

**Logic xử lý chi tiết:**

1. **Chuẩn hóa Unicode (NFC):**
   - Dùng `unicodedata.normalize('NFC', text)` để chuẩn hóa
   - Đảm bảo các ký tự tiếng Việt có dấu được biểu diễn nhất quán
   - Xử lý các ký tự full-width → half-width

2. **Xóa HTML tags:**
   - Dùng regex `re.sub(r'<[^>]+>', '', text)` để loại bỏ toàn bộ thẻ HTML
   - Xử lý các entity HTML phổ biến: `&amp;` → `&`, `&lt;` → `<`, `&gt;` → `>`, `&nbsp;` → ` `
   - Dùng `html.unescape()` để giải mã tất cả HTML entities

3. **Loại bỏ ký tự đặc biệt:**
   - Xóa các emoji, biểu tượng đặc biệt
   - Giữ lại các ký tự tiếng Việt (a-zA-Z, dấu), số, dấu câu cơ bản (`.`, `,`, `!`, `?`, `/`, `\`, `:`, `;`, `_`, `-`, `+`, `=`, `*`, `#`, `@`, `~`, `^`, `%`, `&`, `|`, `<`, `>`, `(`, `)`, `[`, `]`, `{}`)
   - Chuẩn hóa khoảng trắng: loại bỏ multiple spaces, trim đầu/cuối
   - Loại bỏ các URL nhúng trong nội dung bài viết

4. **Xóa dữ liệu base64/ảnh nhúng:**
   - Dữ liệu thô có chứa chuỗi `data:image/gif;base64,...` trong cột `thumbnail_url` và `images`
   - Cần loại bỏ hoặc đánh dấu các trường này

**Công cụ sử dụng:** `re`, `unicodedata`, `html`

```python
# Cấu trúc hàm chính
def normalize_unicode(text):
    """Chuẩn hóa Unicode NFC cho văn bản tiếng Việt"""

def remove_html_tags(text):
    """Loại bỏ toàn bộ thẻ HTML và giải mã HTML entities"""

def remove_special_characters(text):
    """Xóa ký tự đặc biệt, emoji, giữ lại tiếng Việt và số"""

def normalize_whitespace(text):
    """Chuẩn hóa khoảng trắng"""

def normalize_text(text):
    """Pipeline chuẩn hóa đầy đủ cho một đoạn văn bản"""

def normalize_dataframe(df):
    """Chuẩn hóa toàn bộ DataFrame, trả về (df, stats)"""

def visualize_normalize_report(stats):
    """Vẽ 3 biểu đồ trực quan hóa kết quả chuẩn hóa"""
```

**Trực quan hóa kết quả Normalizer (3 biểu đồ):**

| # | File | Mô tả |
|---|------|-------|
| 1 | `normalize_01_noise_types.png` | Bar chart số lượng từng loại nhiễu đã loại bỏ (HTML tags, URL nhúng, Base64, Emoji) |
| 2 | `normalize_02_chars_comparison.png` | Bar chart so sánh tổng ký tự trước/sau chuẩn hóa |
| 3 | `normalize_03_noise_distribution.png` | Pie chart tỷ lệ phân bổ các loại nhiễu đã loại bỏ |

---

## Giai đoạn 3: Xử lý Trùng lặp (Deduplication)

#### [NEW] [deduplicator.py](file:///d:/Dev/Code/Nhom8_DataMining_NEWS/src/deduplicator.py)

**Mục tiêu:** Phát hiện và loại bỏ các bài viết bị lặp lại nhiều lần trong tập dữ liệu.

**Phương pháp:** Sử dụng `pandas.DataFrame.duplicated()` để so sánh trực tiếp giá trị các cột, không sử dụng hash.

**Logic xử lý chi tiết:**

1. **Trùng lặp chính xác theo nội dung (Exact Duplicate):**
   - Dùng `df.duplicated(subset=['main_content'], keep='first')` để tìm các bài viết có nội dung `main_content` giống hệt nhau
   - Giữ lại bản ghi xuất hiện đầu tiên, loại bỏ các bản ghi lặp lại sau đó
   - Log danh sách các bản ghi bị loại bỏ

2. **Trùng lặp theo tiêu đề (Title Duplicate):**
   - Dùng `df.duplicated(subset=['title'], keep='first')` để phát hiện các bài viết cùng tiêu đề
   - Các bài từ nhiều nguồn có thể đăng lại cùng nội dung với tiêu đề giống nhau

3. **Báo cáo trùng lặp:**
   - Thống kê số lượng bản ghi trùng theo từng nguồn (`domain`)
   - Tỷ lệ trùng lặp tổng thể
   - Xóa các trường dữ liệu không cần thiết

**Công cụ sử dụng:** `pandas`

```python
# Cấu trúc hàm chính
def find_exact_duplicates(df, column='main_content'):
    """Phát hiện các bài viết có nội dung bị lặp lại bằng pandas.duplicated()"""

def find_title_duplicates(df, column='title'):
    """Phát hiện các bài viết có tiêu đề bị lặp lại"""

def remove_duplicates(df, subset=['main_content'], keep='first'):
    """Loại bỏ các bài viết trùng lặp, giữ lại bản ghi đầu tiên"""

def generate_dedup_report(df_before, df_after):
    """Tạo báo cáo thống kê về quá trình loại bỏ trùng lặp"""
```

**Trực quan hóa kết quả Deduplication (3 biểu đồ):**

| # | File | Mô tả |
|---|------|-------|
| 1 | `dedup_01_domain_bar.png` | Bar chart số bản ghi theo domain trước/sau loại trùng |
| 2 | `dedup_02_pie_chart.png` | Pie chart tỷ lệ trùng lặp |
| 3 | `dedup_03_comparison.png` | Bar chart so sánh trước/sau dedup |

---

## Giai đoạn 4: Lắp ráp Pipeline hoàn chỉnh

#### [NEW] [pipeline.py](file:///d:/Dev/Code/Nhom8_DataMining_NEWS/src/pipeline.py)

**Mục tiêu:** Kết nối tất cả module của các thành viên thành một pipeline chạy xuyên suốt từ đầu đến cuối.

**Thiết kế Pipeline:**

```
[news_data.csv] 
    → Bước 1: Đọc dữ liệu thô (Thành viên 1)
    → Bước 2: Làm sạch dữ liệu (cleaning.py - Thành viên 2)
    → Bước 3: Chuẩn hóa văn bản (normalizer.py - Thành viên 2)
    → Bước 4: Xử lý trùng lặp (deduplicator.py - Thành viên 2)
    → Bước 5: Tiền xử lý NLP và tạo TF-IDF (preprocessing.py - Thành viên 3)
    → Bước 6: Phân cụm & Gán nhãn (modeling.py - Thành viên 4)
    → Bước 7: Huấn luyện mô hình (Thành viên 5)
    → Bước 8: Xuất dữ liệu & Báo cáo (Thành viên 6)
    → [data/cleaned_news.csv]
    → [data/processed_news.csv] + [data/tfidf_features.pkl] + [models/vectorizer.pkl]
```

**Logic xử lý chi tiết:**

1. **Cấu trúc Pipeline class:**
   - Mỗi bước là một hàm có interface thống nhất: `step(df) → df`
   - Có logging chi tiết cho từng bước (thời gian, số lượng bản ghi vào/ra)
   - Hỗ trợ chạy từng bước riêng lẻ hoặc chạy toàn bộ
   - Lưu checkpoint sau mỗi bước (để có thể resume nếu lỗi)

2. **Cấu hình Pipeline:**
   - File config (JSON/YAML) để điều chỉnh tham số mà không cần sửa code
   - Các tham số: đường dẫn input/output, ngưỡng lọc, chiến lược dedup...

3. **Xử lý lỗi:**
   - Try/catch cho từng bước
   - Ghi log lỗi chi tiết
   - Cho phép skip bước bị lỗi hoặc dừng toàn bộ

4. **Tích hợp bước tiền xử lý NLP của Thành viên 3:**
   - Nhận dữ liệu đã qua làm sạch, chuẩn hóa và loại trùng từ các bước trước.
   - Đọc cột nội dung chính `main_content`.
   - Tạo cột `processed_content` sau khi chuyển chữ thường, loại bỏ ký tự không cần thiết và lọc stopwords tiếng Việt.
   - Trích xuất đặc trưng văn bản bằng TF-IDF.
   - Lưu dữ liệu đã tiền xử lý vào `data/processed_news.csv`.
   - Lưu ma trận đặc trưng vào `data/tfidf_features.pkl`.
   - Lưu bộ vectorizer đã fit vào `models/vectorizer.pkl` để tái sử dụng khi phân cụm, huấn luyện hoặc dự đoán dữ liệu mới.

**Tóm tắt cho slide bước NLP:**

```
cleaned_news.csv
    → lowercase + lọc ký tự nhiễu
    → loại stopwords tiếng Việt
    → tạo cột processed_content
    → TF-IDF vectorization
    → processed_news.csv + tfidf_features.pkl + vectorizer.pkl
```

```python
# Cấu trúc chính
class DataPipeline:
    def __init__(self, config_path='config.json'):
        """Khởi tạo pipeline với cấu hình"""
    
    def step_load_data(self):
        """Bước 1: Đọc dữ liệu thô"""
    
    def step_clean(self):
        """Bước 2: Làm sạch"""
    
    def step_normalize(self):
        """Bước 3: Chuẩn hóa"""
    
    def step_deduplicate(self):
        """Bước 4: Loại trùng lặp"""
    
    def step_preprocess_nlp(self):
        """Bước 5: Tiền xử lý NLP (gọi code Thành viên 3)"""
    
    def step_clustering(self):
        """Bước 6: Phân cụm (gọi code Thành viên 4)"""
    
    def run_all(self):
        """Chạy toàn bộ pipeline"""
    
    def run_step(self, step_name):
        """Chạy một bước cụ thể"""

    def _visualize_pipeline_funnel(self):
        """Vẽ biểu đồ Waterfall/Funnel tổng thể pipeline"""

    # Chạy theo từng thành viên
    def run_tv1(self): """TV1: Thu thập & EDA"""
    def run_tv2(self): """TV2: Làm sạch, Chuẩn hóa, Loại trùng"""
    def run_tv3(self): """TV3: NLP & TF-IDF (đọc cleaned_news.csv)"""
    def run_tv4(self): """TV4: Phân cụm & Gán nhãn (đọc processed_news.csv)"""
    def run_tv5(self): """TV5: Deep Learning & Tuning"""
    def run_tv6(self): """TV6: Xuất dữ liệu & Báo cáo"""
```

**Chạy pipeline theo từng thành viên (CLI):**

```bash
# Chạy toàn bộ pipeline
python src/pipeline.py all

# Chạy riêng từng thành viên
python src/pipeline.py tv1    # TV1: Thu thập dữ liệu & EDA
python src/pipeline.py tv2    # TV2: Làm sạch, Chuẩn hóa, Loại trùng
python src/pipeline.py tv3    # TV3: Tiền xử lý NLP & TF-IDF
python src/pipeline.py tv4    # TV4: Phân cụm & Gán nhãn
python src/pipeline.py tv5    # TV5: Deep Learning & Tuning
python src/pipeline.py tv6    # TV6: Xuất dữ liệu & Báo cáo
```

> [!NOTE]
> - TV3 yêu cầu đã chạy TV2 trước (cần file `cleaned_news.csv`)
> - TV4 yêu cầu đã chạy TV3 trước (cần file `processed_news.csv`)

**Trực quan hóa Pipeline tổng thể (1 biểu đồ):**

| # | File | Mô tả |
|---|------|-------|
| 1 | `pipeline_funnel.png` | Waterfall chart thể hiện số bản ghi giảm dần qua từng bước pipeline (Dữ liệu thô → Làm sạch → Chuẩn hóa → Loại trùng → NLP) |

#### [NEW] [config.json](file:///d:/Dev/Code/Nhom8_DataMining_NEWS/config.json)

```json
{
    "input_path": "data/news_data.csv",
    "output_cleaned_path": "data/cleaned_news.csv",
    "output_processed_path": "data/processed_news.csv",
    "cleaning": {
        "required_columns": ["title", "main_content", "url"],
        "min_words": 50,
        "max_words": 10000
    },
    "deduplication": {
        "column": "main_content",
        "keep": "first"
    },
    "preprocessing": {
        "stopword_path": "data/vietnamese-stopwords.txt",
        "content_column": "main_content",
        "processed_column": "processed_content",
        "tfidf_matrix_path": "data/tfidf_features.pkl",
        "vectorizer_path": "models/vectorizer.pkl",
        "n_components": null,
        "max_features": null
    },
    "encoding": "utf-8-sig"
}
```

---

## Cấu trúc file sau khi hoàn thành

```
src/
├── cleaning.py        # [MỚI] Module làm sạch dữ liệu (TV2)
├── normalizer.py      # [MỚI] Module chuẩn hóa Unicode/HTML (TV2)
├── deduplicator.py    # [MỚI] Module xử lý trùng lặp (TV2)
├── pipeline.py        # [MỚI] Pipeline tổng hợp (TV2)
├── preprocessing.py   # Module NLP + TF-IDF (TV3)
├── modeling.py        # Module ML (TV4/5 - chờ code)
└── predict.py         # Module dự đoán (TV5 - chờ code)

config.json            # [MỚI] File cấu hình pipeline
data/
├── news_data.csv      # Dữ liệu thô (~19.120 bản ghi)
├── cleaned_news.csv   # [SẼ TẠO] Dữ liệu sau khi làm sạch
├── processed_news.csv # [SẼ TẠO] Dữ liệu có thêm cột processed_content
└── tfidf_features.pkl # [SẼ TẠO] Ma trận đặc trưng TF-IDF

models/
└── vectorizer.pkl     # [SẼ TẠO] TF-IDF vectorizer đã fit
```

---

## Thứ tự triển khai đề xuất

| Thứ tự | Công việc | Ước lượng thời gian | Phụ thuộc |
|--------|----------|-------------------|-----------|
| 1 | Tạo `cleaning.py` - Làm sạch dữ liệu | 2-3 giờ | Cần `news_data.csv` (đã có) |
| 2 | Tạo `normalizer.py` - Chuẩn hóa văn bản | 2-3 giờ | Không phụ thuộc |
| 3 | Tạo `deduplicator.py` - Xử lý trùng lặp | 1-2 giờ | Cần bước 1, 2 hoàn thành |
| 4 | Tạo `config.json` - File cấu hình | 30 phút | Không phụ thuộc |
| 5 | Tạo `pipeline.py` - Lắp ráp pipeline | 3-4 giờ | Cần bước 1-4, cần code các TV khác |
| 6 | Kiểm thử & chạy thử pipeline | 2-3 giờ | Cần bước 5 |

**Tổng thời gian ước lượng:** ~11-16 giờ làm việc

---

## Thư viện cần thêm vào `requirements.txt`

```
pandas
unicodedata  # (thư viện built-in, không cần cài)
```

> [!IMPORTANT]
> **`unicodedata`, `re`, `html`** là các thư viện built-in của Python, không cần cài thêm. Chỉ cần cài `pandas` nếu chưa có.

---

## Kế hoạch Kiểm thử

### Automated Tests
- Chạy `python src/pipeline.py` để kiểm tra pipeline chạy xuyên suốt
- Kiểm tra file output `data/cleaned_news.csv` được tạo thành công
- So sánh số lượng bản ghi trước/sau mỗi bước

### Kiểm tra thủ công
- Mở file CSV đầu ra để kiểm tra dữ liệu đã sạch
- Kiểm tra không còn HTML tags, ký tự lỗi trong nội dung
- Kiểm tra không còn bản ghi trùng lặp

---

## Open Questions

> [!NOTE]
> 1. **Encoding file CSV:** Dữ liệu hiện tại có vấn đề encoding khi đọc bằng PowerShell. Cần xác nhận encoding thực tế (UTF-8, UTF-8 BOM, hay encoding khác) để xử lý đúng.
> 2. **Ngưỡng lọc bài viết:** Min 50 từ, max 10.000 từ có phù hợp không? Có muốn điều chỉnh?
> 3. **Chiến lược giữ bản ghi trùng:** Khi phát hiện trùng lặp, giữ bản ghi đầu tiên hay bản ghi mới nhất?
> 4. **Cần lắp ráp code của các thành viên khác ngay không?** Hay chỉ cần tạo interface/placeholder cho pipeline trước, rồi tích hợp sau khi có code?

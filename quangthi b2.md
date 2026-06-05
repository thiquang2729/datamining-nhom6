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
| 3 | Xử lý trùng lặp bằng MD5 Hash | `src/deduplicator.py` | ⭐⭐ |
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
   - Giữ lại các ký tự tiếng Việt (a-zA-Z, dấu), số, dấu câu cơ bản (`.`, `,`, `!`, `?`)
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
```

---

## Giai đoạn 3: Xử lý Trùng lặp (Deduplication) bằng MD5 Hash

#### [NEW] [deduplicator.py](file:///d:/Dev/Code/Nhom8_DataMining_NEWS/src/deduplicator.py)

**Mục tiêu:** Phát hiện và loại bỏ các bài viết trùng lặp nội dung.

**Logic xử lý chi tiết:**

1. **Trùng lặp chính xác (Exact Duplicate):**
   - Tính MD5 hash cho cột `main_content` (sau khi đã chuẩn hóa)
   - Nhóm các bản ghi có cùng hash → giữ lại bản ghi đầu tiên (hoặc bản ghi mới nhất theo `published_time`)
   - Log danh sách các bản ghi bị loại bỏ

2. **Trùng lặp gần đúng (Near Duplicate) - Tùy chọn:**
   - Tính MD5 hash cho `title` để phát hiện các bài viết cùng tiêu đề nhưng khác nội dung nhẹ
   - Các bài từ nhiều nguồn có thể đăng lại cùng nội dung với thay đổi nhỏ

3. **Báo cáo trùng lặp:**
   - Thống kê số lượng bản ghi trùng theo từng nguồn (`domain`)
   - Tỷ lệ trùng lặp tổng thể

**Công cụ sử dụng:** `hashlib`, `pandas`

```python
# Cấu trúc hàm chính
def compute_md5(text):
    """Tính MD5 hash cho một chuỗi văn bản"""

def find_exact_duplicates(df, column='main_content'):
    """Phát hiện các bản ghi trùng lặp chính xác dựa trên MD5 hash"""

def find_title_duplicates(df, column='title'):
    """Phát hiện trùng lặp dựa trên tiêu đề"""

def remove_duplicates(df, keep='first'):
    """Loại bỏ trùng lặp, giữ lại bản ghi theo chiến lược chọn"""

def generate_dedup_report(df_before, df_after):
    """Tạo báo cáo thống kê về quá trình loại bỏ trùng lặp"""
```

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
    → Bước 5: Tiền xử lý NLP (preprocessing.py - Thành viên 3)
    → Bước 6: Phân cụm & Gán nhãn (modeling.py - Thành viên 4)
    → Bước 7: Huấn luyện mô hình (Thành viên 5)
    → Bước 8: Xuất dữ liệu & Báo cáo (Thành viên 6)
    → [data/cleaned_news.csv] + [data/processed_news.csv]
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
```

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
        "method": "md5",
        "column": "main_content",
        "keep": "first"
    },
    "encoding": "utf-8"
}
```

---

## Cấu trúc file sau khi hoàn thành

```
src/
├── cleaning.py        # [MỚI] Module làm sạch dữ liệu (TV2)
├── normalizer.py      # [MỚI] Module chuẩn hóa Unicode/HTML (TV2)
├── deduplicator.py    # [MỚI] Module xử lý trùng lặp MD5 (TV2)
├── pipeline.py        # [MỚI] Pipeline tổng hợp (TV2)
├── preprocessing.py   # Module NLP (TV3 - chờ code)
├── modeling.py        # Module ML (TV4/5 - chờ code)
└── predict.py         # Module dự đoán (TV5 - chờ code)

config.json            # [MỚI] File cấu hình pipeline
data/
├── news_data.csv      # Dữ liệu thô (~19.120 bản ghi)
├── cleaned_news.csv   # [SẼ TẠO] Dữ liệu sau khi làm sạch
└── processed_news.csv # [SẼ TẠO] Dữ liệu sau toàn bộ pipeline
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
hashlib  # (thư viện built-in, không cần cài)
unicodedata  # (thư viện built-in, không cần cài)
```

> [!IMPORTANT]
> **`hashlib`, `unicodedata`, `re`, `html`** là các thư viện built-in của Python, không cần cài thêm. Chỉ cần cài `pandas` nếu chưa có.

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

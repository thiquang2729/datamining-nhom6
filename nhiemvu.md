# Phân chia nv

### **Thành viên 1: Ngô Hoàng Anh**

**Data Engineer (Thu thập & Lưu trữ)**

* Cào dữ liệu (crawl) khoảng 10.000 \- 15.000 bài báo chuyên mục Công nghệ từ các trang web (Tinhte, VnExpress Số Hóa, GenK, Thanh Niên...).  
* Trích xuất các trường cơ bản: Tiêu đề, Nội dung, URL, Ngày đăng.  
* Lưu trữ dữ liệu thô vào cơ sở dữ liệu để dễ dàng truy xuất.  
* Vẽ biểu đồ thống kê cơ bản EDA (số lượng bài theo nguồn, độ dài trung bình của các bài viết).  
* **Công cụ:** `requests`, `BeautifulSoup` / `Selenium` (để cào HTML), `MongoDB` / `pymongo` (lưu trữ), `matplotlib`, `seaborn`.  
* **Kỹ năng:** Web Scraping, NoSQL, Data Understanding.

### **Thành viên 2: Lê Quang Thi**

**Data Quality & Pipeline Engineer (Làm sạch & Tự động hóa)**

* Loại bỏ các **line** dữ liệu rỗng, URL lỗi, bài viết không lấy được chữ.  
* Xóa HTML tags, ký tự đặc biệt, chuẩn hóa Unicode (NFC).  
* Lọc các **line** bài viết quá ngắn (\<50 từ) hoặc quá dài.  
* Xử lý dữ liệu trùng lặp (Duplicate) bằng thuật toán Hash MD5.  
* Lắp ráp code của các thành viên khác thành một Pipeline chạy xuyên suốt từ đầu đến cuối.  
* **Công cụ:** `Python` (xử lý chuỗi), `re` (Regex), `unicodedata`, `hashlib`.  
* **Kỹ năng:** Data Cleaning (Xử lý nhiễu, missing), Biểu thức chính quy, Kỹ năng thiết kế luồng hệ thống.

### **Thành viên 3: Tôn Hoàng Nhớ**

### **Feature Engineer (Tiền xử lý NLP & Trích xuất đặc trưng)**

* Chuyển văn bản về chữ thường (Lowercasing).  
* Tách từ tiếng Việt (Tokenization) cho từng **line** văn bản (ví dụ: ghép "trí tuệ\_nhân tạo").  
* Lọc bỏ Stopwords (các từ vô nghĩa như "và", "hoặc", "là").  
* Chuyển các **line** văn bản đã sạch thành Vector số học bằng TF-IDF hoặc Word Embedding (Word2Vec).  
* Giảm chiều dữ liệu nếu ma trận đặc trưng quá lớn.  
* **Công cụ:** `VnCoreNLP` / `pyvi` / `Underthesea`, `scikit-learn` (TfidfVectorizer, TruncatedSVD), `Gensim`.  
* **Kỹ năng:** Xử lý ngôn ngữ tự nhiên (NLP) tiếng Việt, Khái niệm ma trận thưa (Sparse Matrix), Giảm chiều dữ liệu.

### **Thành viên 4: Nguyễn Văn Trường**

**ML Engineer 1 (Clustering & Gán nhãn Lab)**

* Tiếp nhận ma trận dữ liệu từ người số 3 (lúc này chưa có nhãn).  
* Áp dụng thuật toán phân cụm không giám sát (K-Means, DBSCAN) để chia các **line** bài báo thành các cụm (cluster) có chung đặc điểm từ vựng.  
* Phân tích từ khóa trung tâm của từng cụm (ví dụ: cụm 1 toàn từ CPU, RAM; cụm 2 toàn mã độc, hacker).  
* Thực hiện gán nhãn thủ công (Phần cứng, An ninh mạng, AI...) cho các cụm này để tạo ra tập dữ liệu gốc (Ground Truth).  
* Chia tập dữ liệu đã có nhãn thành Train/Val/Test đảm bảo cân bằng class (Stratified Split).  
* **Công cụ:** `scikit-learn` (KMeans, DBSCAN, Silhouette Score).  
* **Kỹ năng:** Học không giám sát (Unsupervised Learning), Phân tích đặc trưng văn bản, Kỹ năng phân loại domain Công nghệ.

### **Thành viên 5: (Tên thành viên 5 \- Trọng tâm AI)**

**ML Engineer 2 (Deep Learning & Tuning)**

* Thiết kế và xây dựng mô hình Deep Learning (như yêu cầu bắt buộc của thầy) dùng mạng CNN cho văn bản hoặc LSTM.  
* Tinh chỉnh siêu tham số (Hyperparameter Tuning): Learning rate, batch size, số lớp ẩn (hidden layers).  
* **Công cụ:** `TensorFlow` / `Keras` hoặc `PyTorch`, `scikit-learn` (train\_test\_split, classification\_report).  
* **Kỹ năng:** Học sâu (Deep Learning), Xử lý mất cân bằng dữ liệu, Đánh giá mô hình (Precision, Recall, F1).

### **Thành viên 6: Khánh Huyền**

**Technical Writer & Data Exporter (Document & Slide)**

* Đánh giá chất lượng mô hình qua các chỉ số Accuracy, F1-Score, vẽ Confusion Matrix.  
* Viết báo cáo chuẩn định dạng CRISP-DM, mô tả rõ bài toán từ Business Understanding đến khâu ra dữ liệu cuối.  
* Đóng gói toàn bộ tập dữ liệu đầu ra cuối cùng (đã được dán nhãn chuẩn, sạch sẽ) thành file JSON/CSV \- đây chính là "Output là dữ liệu" để nộp cho thầy.  
* Thiết kế Slide thuyết trình, vẽ các sơ đồ Pipeline trực quan.  
* Quản lý kho mã nguồn GitHub chung của nhóm, cập nhật file `README.md` và `requirements.txt`.  
* **Công cụ:** `Microsoft Word/PowerPoint`, `Git/GitHub`, `Draw.io` / `Figma` (vẽ sơ đồ).  
* **Kỹ năng:** Hiểu toàn bộ vòng đời phân tích dữ liệu, Kỹ năng tổng hợp thông tin, Trình bày trực quan.

# Doc
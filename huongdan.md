# 📝 Hướng Dẫn Phân Công Công Việc - Nhóm 8 Data Mining

Tài liệu hướng dẫn phân chia nhiệm vụ và trách nhiệm của từng thành viên cho các thư mục và tệp tin trong dự án.

---

### 📂 1. `data/`
* **Thành viên phụ trách chính:**
  * 👤 Thành viên 1 (Thu thập)
  * 👤 Thành viên 2 (Làm sạch)
* **Mô tả công việc:**
  * Chứa các file dữ liệu: `raw_data.json`, `train_data.json`, `unlabeled_data.json`.
  * ⚠️ **Lưu ý**: Không push file dữ liệu hàng chục MB lên GitHub để tránh làm nặng repository.

---

### 📄 2. `src/preprocessing.py`
* **Thành viên phụ trách chính:**
  * 👤 Thành viên 2 (Làm sạch)
  * 👤 Thành viên 3 (NLP & Feature)
* **Mô tả công việc:**
  * Viết các hàm loại bỏ HTML, Outlier, Duplicate.
  * Viết pipeline chuẩn hóa văn bản (Lowercasing, Tokenization, Xóa Stopword).
  * Viết hàm TF-IDF và SVD để biến đổi dữ liệu.

---

### 📄 3. `src/modeling.py`
* **Thành viên phụ trách chính:**
  * 👤 Thành viên 4 (Base Model)
  * 👤 Thành viên 5 (Self-Training)
* **Mô tả công việc:**
  * **Thành viên 4**: Viết code chia tập Stratified Split và huấn luyện mô hình Logistic Regression cơ sở.
  * **Thành viên 5**: Viết vòng lặp tự gán nhãn (Self-training) và tính toán các chỉ số đánh giá F1-score, Accuracy.

---

### 📂 4. `models/`
* **Thành viên phụ trách chính:**
  * 👤 Thành viên 5 (Đánh giá & Lưu model)
* **Mô tả công việc:**
  * Lưu trữ các file đầu ra như: `vectorizer.pkl`, `final_model_calibrated.pkl`.
  * Lưu bộ nhãn giả `pseudo_labels.json` để phục vụ công việc của cả nhóm.

---

### 📄 5. `demo/app.py` & `src/predict.py`
* **Thành viên phụ trách chính:**
  * 👤 Thành viên 6 (Technical Writer & Deployer)
* **Mô tả công việc:**
  * Gọi file mô hình đã lưu từ thư mục `models/` để đưa vào giao diện.
  * Xây dựng giao diện người dùng (UI) bằng Streamlit cho phép nhập văn bản trực tiếp và dự đoán, in ra chủ đề.

---

### 📄 6. `notebooks/notebook.ipynb`
* **Thành viên phụ trách chính:**
  * 👤 Thành viên 1 (EDA)
  * 👤 Thành viên 5 (Evaluation)
* **Mô tả công việc:**
  * Thư mục dùng chung làm không gian chạy nháp các ý tưởng.
  * **Thành viên 1**: Phân tích dữ liệu khám phá, vẽ biểu đồ phân bố bài báo.
  * **Thành viên 5**: Vẽ Confusion Matrix và biểu đồ phân bố độ tin cậy (Confidence).

---

### 📄 7. Các tài liệu & Cấu hình (`README.md`, `requirements.txt`, `report.pdf`)
* **Thành viên phụ trách chính:**
  * 👤 Thành viên 6 (Technical Writer & Deployer)
* **Mô tả công việc:**
  * Tổng hợp báo cáo nghiên cứu định dạng PDF để lưu trữ trong repo (`report.pdf`).
  * Liệt kê danh sách các thư viện cần cài đặt (`pandas`, `scikit-learn`, `pymongo`,...) vào tệp `requirements.txt`.
  * Viết tài liệu hướng dẫn chạy code chi tiết vào tệp `README.md`.
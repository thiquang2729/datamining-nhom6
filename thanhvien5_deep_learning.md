# Ghi chú Thành viên 5 - Deep Learning & Tuning

## Vai trò

Thành viên 5 phụ trách huấn luyện mô hình phân loại văn bản sau khi dữ liệu đã được tiền xử lý và gán nhãn bởi các bước trước.

## File phụ trách

```text
src/deep_learning_model.py
src/predict.py
```

## Dữ liệu đầu vào

| File                    | Nội dung                                                    |
| ----------------------- | ----------------------------------------------------------- |
| `data/train.csv`        | Tập huấn luyện đã có cột `label`                            |
| `data/val.csv`          | Tập validation để kiểm tra mô hình trong quá trình đánh giá |
| `data/test.csv`         | Tập test dùng để đánh giá kết quả cuối                      |
| `models/vectorizer.pkl` | Bộ TF-IDF vectorizer đã được tạo ở bước tiền xử lý NLP      |

## Quy trình thực hiện

1. Đọc dữ liệu `train.csv`, `val.csv`, `test.csv`.
2. Lấy cột văn bản đã xử lý `processed_content`.
3. Lấy nhãn phân loại từ cột `label`.
4. Dùng `models/vectorizer.pkl` để chuyển văn bản thành vector TF-IDF.
5. Mã hóa nhãn bằng `LabelEncoder`.
6. Huấn luyện mô hình `MLPClassifier`.
7. Đánh giá mô hình bằng Accuracy, F1-score, Classification Report và Confusion Matrix.
8. Lưu model, metrics và biểu đồ phục vụ báo cáo.

## Mô hình sử dụng

Mô hình sử dụng trong bước này là `MLPClassifier` của thư viện `scikit-learn`.

Cấu hình chính:

```text
hidden_layer_sizes = (128, 64)
learning_rate_init = 0.001
batch_size = 64
max_iter = 30
early_stopping = True
```

## Kết quả chạy

Kết quả sau khi chạy `python -B src\deep_learning_model.py`:

```text
status: completed
backend: sklearn_mlp
train_size: 12974
val_size: 2780
test_size: 2781
val_accuracy: 0.9540
val_f1_macro: 0.9523
test_accuracy: 0.9522
test_f1_macro: 0.9527
```

Diễn giải ngắn:

```text
Mô hình đạt Accuracy khoảng 95.22% và F1-macro khoảng 95.27% trên tập test.
Kết quả này cho thấy mô hình phân loại văn bản hoạt động ổn định trên các nhóm chủ đề đã được gán nhãn.
```

## File đầu ra

| File                                              | Nội dung                               |
| ------------------------------------------------- | -------------------------------------- |
| `models/deep_learning_text_model.pkl`             | Model đã huấn luyện và label encoder   |
| `models/deep_learning_metrics.json`               | Các chỉ số đánh giá model              |
| `data/deep_learning_confusion_matrix.csv`         | Ma trận nhầm lẫn dạng CSV              |
| `notebooks/deep_learning_01_confusion_matrix.png` | Biểu đồ Confusion Matrix               |
| `notebooks/deep_learning_02_metrics.png`          | Biểu đồ Accuracy và F1-score           |
| `notebooks/deep_learning_03_class_f1.png`         | Biểu đồ F1-score theo từng nhãn        |
| `notebooks/deep_learning_04_dataset_split.png`    | Biểu đồ chia tập Train/Validation/Test |

## Nội dung có thể đưa vào báo cáo

Thành viên 5 xây dựng mô hình phân loại văn bản dựa trên dữ liệu đã được gán nhãn từ bước phân cụm. Dữ liệu đầu vào được biểu diễn bằng TF-IDF, sau đó đưa vào mô hình MLPClassifier để học quan hệ giữa nội dung bài viết và nhãn chủ đề. Mô hình được đánh giá bằng Accuracy, F1-score và Confusion Matrix. Kết quả trên tập test đạt Accuracy 95.22% và F1-macro 95.27%, cho thấy mô hình có khả năng phân loại ổn định giữa các nhóm chủ đề.

## Nội dung có thể đưa vào slide

```text
Input: train/val/test đã gán nhãn + TF-IDF vectorizer
Model: MLPClassifier, 2 hidden layers (128, 64)
Output: model phân loại + metrics + confusion matrix
Kết quả: Test Accuracy 95.22%, Test F1-macro 95.27%
```

## Lệnh chạy

```powershell
cd "D:\datamining-nhom6-main (2)\datamining-nhom6-main"
$env:PYTHONIOENCODING="utf-8"
python -B src\deep_learning_model.py
```

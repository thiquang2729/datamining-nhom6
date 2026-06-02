# Tổng quan đề tài

# **BÁO CÁO KIẾN TRÚC DỰ ÁN KHAI PHÁ DỮ LIỆU: PIPELINE TINH CHẾ VÀ PHÂN LOẠI TIN TỨC CÔNG NGHỆ**

Tài liệu này mô tả chi tiết kiến trúc hệ thống xử lý dữ liệu dạng pipeline, tự động hóa luồng chuyển đổi từ dữ liệu tin tức công nghệ thô thành dữ liệu có cấu trúc phục vụ cho các bài toán tìm kiếm và gợi ý.

## **1\. Mục tiêu và Phạm vi Dự án**

Dự án tập trung xây dựng một hệ thống xử lý luồng dữ liệu liên tục (Data Pipeline). Đầu vào của hệ thống là các tập hợp văn bản thô từ các nguồn báo chí công nghệ, và đầu ra là kho dữ liệu sạch, đã gán nhãn phân loại chuẩn xác. Hệ thống áp dụng các kỹ thuật xử lý ngôn ngữ tự nhiên (NLP) và học sâu (Deep Learning) làm lõi xử lý trung tâm.

## **2\. Xác định các Thể loại Báo chí Công nghệ (Target Categories)**

Để bài toán khai phá dữ liệu đạt hiệu quả cao, phạm vi phân loại được thu hẹp trong 5 danh mục cốt lõi của lĩnh vực công nghệ. Cấu trúc chi tiết được thể hiện qua bảng dưới đây:

| Thể loại (Category) | Từ khóa Đặc trưng (Keywords) | Mô tả Dữ liệu Đầu ra   |
| :---- | :---- | :---- |
| Phần cứng (Hardware) | CPU, GPU, RAM, chip bán dẫn, card đồ họa, bo mạch chủ | Các line dữ liệu liên quan đến linh kiện vật lý và thiết bị phần cứng. |
| Phần mềm & Phát triển (Software) | Hệ điều hành, source code, API, Git, Flutter, ứng dụng di động | Các line dữ liệu về giải pháp phần mềm, lập trình và công cụ phát triển. |
| Trí tuệ nhân tạo (AI & Machine Learning) | Deep Learning, mạng nơ-ron, LLM, PhoBERT, mô hình ngôn ngữ | Dữ liệu tập trung vào các thuật toán, nghiên cứu và ứng dụng thông minh. |
| Thiết bị di động (Mobile Devices) | Smartphone, iOS, Android, Xiaomi, màn hình OLED, dung lượng pin | Thông tin đánh giá, cấu hình và thị trường thiết bị di động. |
| An ninh mạng (Cybersecurity) | Mã hóa, mã độc, lừa đảo trực tuyến, tường lửa, lỗ hổng bảo mật | Các bài viết phân tích về lỗ hổng, mã độc và giải pháp bảo mật thông tin. |

## **3\. Kiến trúc Pipeline Xử lý Dữ liệu chi tiết**

Hệ thống vận hành theo một chuỗi các công đoạn nối tiếp nhau, đảm bảo mỗi line dữ liệu đi qua đều được chuẩn hóa và làm giàu thông tin:

### **3.1. Thu thập dữ liệu thô (Input Data)**

* Cào dữ liệu tự động từ các nguồn báo điện tử uy tín về công nghệ tại Việt Nam.  
* Kết quả thu về là các line văn bản thô chứa tiêu đề, nội dung, ngày xuất bản và URL gốc, chưa có bất kỳ nhãn phân loại nào.

### **3.2. Tiền xử lý Ngôn ngữ tự nhiên (NLP Preprocessing)**

Đây là giai đoạn tinh lọc để loại bỏ nhiễu văn bản. Các kỹ thuật NLP cốt lõi áp dụng cho từng line bài viết bao gồm:

1. **Chuẩn hóa văn bản:** Chuyển toàn bộ chữ về dạng in thường, loại bỏ các ký tự đặc biệt, dấu câu thừa và các thẻ HTML phát sinh trong quá trình cào dữ liệu.  
2. **Tách từ tiếng Việt (Tokenization):** Sử dụng các công cụ chuyên dụng cho tiếng Việt như VnCoreNLP hoặc Underthesea để ghép các từ ghép lại với nhau (ví dụ: "trí tuệ\_nhân tạo", "phần\_mềm").  
3. **Loại bỏ từ dừng (Stopwords):** Lọc bỏ các từ xuất hiện phổ biến nhưng không mang giá trị phân loại như "và", "hoặc", "nhưng", "tại", "với".  
4. **Trích xuất đặc trưng (Feature Extraction):** Biến đổi các line văn bản sau khi làm sạch thành các vector số học thông qua phương pháp TF-IDF hoặc nhúng từ (Word Embedding).

### **3.3. Phân cụm và Khám phá Cấu trúc (Unsupervised Clustering)**

Do dữ liệu ban đầu hoàn toàn chưa có nhãn, hệ thống cần một bước khám phá tự động:

* Áp dụng thuật toán phân cụm K-Means để tự động nhóm các line bài viết có độ tương đồng cao về mặt từ vựng vào chung một phân vùng.  
* Mục tiêu là tìm ra ranh giới tự nhiên giữa các chủ đề lớn trong tập dữ liệu.

### **3.4. Chuẩn hóa Nhãn hệ thống (Lab Labeling)**

* Kiểm tra chéo (Double check) các phân vùng dữ liệu đã được gom cụm.  
* Gán nhãn đại diện tương ứng (Phần cứng, Phần mềm, AI...) cho các nhóm bài viết để tạo ra tập dữ liệu huấn luyện chuẩn (Ground Truth).

### **3.5. Huấn luyện Mô hình Học sâu (Deep Learning Pipeline)**

* Sử dụng tập dữ liệu đã gán nhãn để huấn luyện các kiến trúc mạng nơ-ron như CNN cho văn bản hoặc tinh chỉnh (Fine-tuning) mô hình ngôn ngữ lớn chuyên dụng cho tiếng Việt (PhoBERT).  
* Mô hình này đóng vai trò tự động hóa việc phân loại cho mọi line dữ liệu mới phát sinh trong tương lai mà không cần can thiệp thủ công.

## **4\. Định dạng Dữ liệu Đầu ra (Output Data) và Hướng Ứng dụng**

Đầu ra cuối cùng của toàn bộ pipeline là một kho dữ liệu đã được cấu trúc hóa toàn diện dưới dạng JSON hoặc lưu trữ trực tiếp trong cơ sở dữ liệu. Mỗi line bản ghi dữ liệu sẽ bao gồm các trường thông tin sạch:

{  
  "article\_id": "TECH\_2026\_001",  
  "title": "Đánh giá hiệu năng xử lý của thế hệ vi xử lý mới",  
  "cleaned\_content": "đánh\_giá hiệu\_năng xử\_lý thế\_hệ vi\_xử\_lý chip hiệu\_suất phần\_cứng",  
  "assigned\_category": "Phần cứng",  
  "processed\_at": "2026-05-31T14:05:00Z"  
}

**Ứng dụng thực tế:** Kho dữ liệu đầu ra có cấu trúc này sẽ trực tiếp làm nền tảng để xây dựng công cụ bộ lọc tìm kiếm nâng cao hoặc hệ thống gợi ý bài viết liên quan (Recommendation System) dựa trên mức độ tương đồng của nhãn và từ khóa đặc trưng giữa các bài viết.




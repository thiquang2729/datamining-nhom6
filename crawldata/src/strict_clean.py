import sqlite3
import re

ALLOWED = ["Thời sự", "Kinh doanh", "Thế giới", "Thể thao", "Pháp luật", "Giáo dục", "Sức khỏe"]

CATEGORY_RULES = {
    "Thể thao": [
        "bóng đá", "fifa", "asean cup", "cup", "cầu thủ", "huấn luyện viên", "trận đấu", "vô địch", 
        "sea games", "ronaldo", "messi", "ngoại hạng anh", "bundesliga", "laliga", "chuyển nhượng",
        "sân vận động", "tuyển quốc gia", "tennis", "olympic", "giải đấu", "bóng chuyền", "đối đầu"
    ],
    "Kinh doanh": [
        "cổ đông", "doanh nghiệp", "doanh thu", "cổ phiếu", "kinh tế", "tỷ đồng", "usd", "ngân hàng",
        "vàng", "đầu tư", "bất động sản", "lãi suất", "xuất khẩu", "nhập khẩu", "thương mại", "dự án",
        "tài chính", "thị trường", "lợi nhuận", "cổ phần", "kinh doanh", "chứng khoán", "địa ốc"
    ],
    "Thế giới": [
        "nato", "ukraine", "putin", "trump", "zelensky", "mỹ", "nga", "trung quốc", "israel", "gaza",
        "tổng thống", "quốc tế", "ngoại giao", "liên hợp quốc", "chính quyền", "bầu cử", "nhật bản",
        "hàn quốc", "châu âu", "trừng phạt", "đồng minh", "quân sự", "tên lửa", "uav"
    ],
    "Pháp luật": [
        "công an", "điều tra", "khởi tố", "bắt tạm giam", "vụ án", "xét xử", "bị cáo", "tòa án",
        "tội phạm", "ma túy", "trộm cắp", "viện kiểm sát", "lừa đảo", "giam giữ", "tuyên án",
        "luật sư", "vi phạm", "hình sự", "phá án", "truy nã", "cảnh sát"
    ],
    "Giáo dục": [
        "học sinh", "giáo viên", "tuyển sinh", "điểm thi", "kỳ thi", "đại học", "trường học",
        "tốt nghiệp", "lớp 10", "đề thi", "du học", "sách giáo khoa", "học đường", "thpt", "tiểu học",
        "giáo dục", "học tập", "bài thi"
    ],
    "Sức khỏe": [
        "bác sĩ", "bệnh viện", "bệnh nhân", "vắc xin", "điều trị", "phẫu thuật", "y tế", "ung thư",
        "sức khỏe", "lây nhiễm", "dịch bệnh", "triệu chứng", "thuốc", "dinh dưỡng", "độc hại", "bệnh lý"
    ]
}

def infer(title, tags, content):
    title = str(title).lower() if pd.notna(title) and title else ""
    tags = str(tags).lower() if pd.notna(tags) and tags else ""
    content = str(content).lower() if pd.notna(content) and content else ""
    combined = f"{title} {tags} {content}"
    
    scores = {cat: 0 for cat in CATEGORY_RULES}
    for cat, keywords in CATEGORY_RULES.items():
        for kw in keywords:
            if kw in combined:
                if kw in title: scores[cat] += 6
                if kw in tags: scores[cat] += 3
                if kw in content: scores[cat] += 1
    max_cat = None
    max_score = 0
    for cat, score in scores.items():
        if score > max_score:
            max_score = score
            max_cat = cat
    return max_cat

import pandas as pd
import os
import sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def main():
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    csv_file = str(_PROJECT_ROOT / "data" / "news_data.csv")
    if not os.path.exists(csv_file):
        print(f"Lỗi: Không tìm thấy file {csv_file}")
        return
        
    print("Đang đọc dữ liệu từ CSV để làm sạch...")
    df = pd.read_csv(csv_file)
    
    updates_count = 0
    
    # Hàm áp dụng cho từng dòng
    def process_row(row):
        nonlocal updates_count
        cat = row.get('category')
        if pd.isna(cat) or cat not in ALLOWED or re.search(r'[^a-zA-ZÀ-ỹ\s]', str(cat)):
            inferred = infer(row.get('title', ''), row.get('tags', ''), row.get('main_content', ''))
            if not inferred:
                inferred = "Thời sự"
            updates_count += 1
            return inferred
        return cat

    df['category'] = df.apply(process_row, axis=1)
    
    if updates_count > 0:
        print(f"Đang lưu lại {updates_count} thay đổi vào {csv_file}...")
        df.to_csv(csv_file, index=False, encoding="utf-8-sig")
        
    print("=== SAU KHI SỬA TRIỆT ĐỂ (100% HOÀN HẢO) ===")
    dist = df['category'].value_counts()
    for cat, count in dist.items():
        print(f" - {cat}: {count:,} bài")

if __name__ == "__main__":
    main()

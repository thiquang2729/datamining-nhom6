import sys, io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import ast
import numpy as np
import warnings

# Tắt cảnh báo
warnings.filterwarnings("ignore")

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import matplotlib
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']

CSV_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "news_data.csv"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False
    print("Thư viện wordcloud chưa được cài đặt. Bỏ qua vẽ WordCloud.")

def load_data():
    print(f"Đang đọc dữ liệu từ {CSV_PATH.name}...")
    df = pd.read_csv(CSV_PATH)
    return df

def plot_top_tags(df):
    """Trực quan hóa top 20 tags phổ biến nhất"""
    print("1. Đang vẽ biểu đồ Top 20 Tags...")
    all_tags = []
    
    # Xử lý tags: nó có thể là chuỗi dạng list '["tag1", "tag2"]' hoặc chuỗi cách nhau dấu phẩy
    for tags_str in df['tags'].dropna():
        try:
            # Thử parse string dạng list
            if tags_str.startswith('[') and tags_str.endswith(']'):
                tags = ast.literal_eval(tags_str)
            else:
                tags = [t.strip() for t in tags_str.split(',')]
                
            all_tags.extend([str(t).lower() for t in tags if str(t).strip()])
        except Exception:
            continue
            
    if not all_tags:
        print("Không có dữ liệu tags hợp lệ.")
        return

    tag_series = pd.Series(all_tags)
    top_tags = tag_series.value_counts().head(20)

    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x=top_tags.values, y=top_tags.index, palette='viridis')
    
    # Thêm số lượng cụ thể vào cuối mỗi thanh biểu đồ bằng tay để đảm bảo hiển thị 100%
    for i, v in enumerate(top_tags.values):
        ax.text(v + 15, i, f'{v:,}', color='black', va='center', fontsize=11, fontweight='bold')
        
    # Nới rộng trục X thêm 10% để không bị lẹm mất chữ số ở thanh dài nhất
    plt.xlim(0, max(top_tags.values) * 1.1)
    
    plt.title('Top 20 Tags phổ biến nhất trong các bài báo Công nghệ', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Tần suất xuất hiện', fontsize=12)
    plt.ylabel('Tags', fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'advanced_top_tags.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   -> Đã lưu: advanced_top_tags.png")
    
    return all_tags

def plot_wordcloud(all_tags):
    """Vẽ WordCloud cho Tags"""
    if not HAS_WORDCLOUD or not all_tags:
        return
        
    print("2. Đang vẽ WordCloud cho Tags...")
    
    # Tính tần suất thay vì nối chuỗi để giữ nguyên các cụm từ có dấu cách
    from collections import Counter
    frequencies = Counter(all_tags)
    
    # Cấu hình font chữ hỗ trợ tiếng Việt (thường có trên Windows)
    import os
    font_path = "C:/Windows/Fonts/arial.ttf"
    if not os.path.exists(font_path):
        font_path = None # Dùng mặc định nếu không tìm thấy
        
    wordcloud = WordCloud(width=1600, height=1000, 
                         background_color='white', 
                         colormap='viridis',
                         max_words=100,
                         font_path=font_path,
                         contour_width=1,
                         contour_color='steelblue').generate_from_frequencies(frequencies)
                         
    plt.figure(figsize=(16, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('WordCloud: Các chủ đề công nghệ được quan tâm nhất', fontsize=20, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'advanced_tags_wordcloud.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   -> Đã lưu: advanced_tags_wordcloud.png")

def plot_content_length(df):
    """Trực quan hóa độ dài nội dung (Word Count)"""
    print("3. Đang phân tích độ dài bài viết...")
    
    # Tính số từ cho mỗi bài viết
    df['word_count'] = df['main_content'].fillna('').apply(lambda x: len(str(x).split()))
    
    # Bỏ qua các bài không có nội dung thực tế (word_count < 10)
    df_valid = df[df['word_count'] >= 10]
    
    if df_valid.empty:
        print("Không có bài viết nào có nội dung hợp lệ để phân tích độ dài.")
        return
        
    # a. Histogram độ dài bài viết
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df_valid, x='word_count', bins=50, kde=True, color='#2c7bb6')
    
    # Thêm đường line cho trung bình và trung vị
    mean_wc = df_valid['word_count'].mean()
    median_wc = df_valid['word_count'].median()
    plt.axvline(mean_wc, color='red', linestyle='dashed', linewidth=2, label=f'Trung bình: {mean_wc:.0f} từ')
    plt.axvline(median_wc, color='orange', linestyle='dashed', linewidth=2, label=f'Trung vị: {median_wc:.0f} từ')
    
    plt.title('Phân phối độ dài bài báo (Số lượng từ)', fontsize=15, fontweight='bold', pad=15)
    plt.xlabel('Số lượng từ (Word Count)', fontsize=12)
    plt.ylabel('Số bài báo', fontsize=12)
    plt.xlim(0, df_valid['word_count'].quantile(0.95)) # Cắt ở 95% để tránh outliers làm hỏng biểu đồ
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'advanced_wordcount_dist.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   -> Đã lưu: advanced_wordcount_dist.png")
    
    # b. Boxplot so sánh độ dài giữa các domain
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_valid, x='domain', y='word_count', palette='Set2')
    plt.title('So sánh độ dài bài viết giữa các trang báo', fontsize=15, fontweight='bold', pad=15)
    plt.xlabel('Trang báo', fontsize=12)
    plt.ylabel('Số lượng từ', fontsize=12)
    plt.ylim(0, df_valid['word_count'].quantile(0.95)) # Cắt bỏ outliers phía trên cho dễ nhìn
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'advanced_wordcount_domain.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   -> Đã lưu: advanced_wordcount_domain.png")

def plot_missing_values(df):
    """Trực quan hóa tỷ lệ dữ liệu bị thiếu"""
    print("4. Đang phân tích chất lượng dữ liệu (Missing Values)...")
    
    # Lọc bỏ các cột trống hoàn toàn (100% missing) do không sử dụng
    df_used = df.dropna(axis=1, how='all')

    # Tính phần trăm missing
    missing_pct = (df_used.isnull().sum() / len(df_used)) * 100
    
    plt.figure(figsize=(12, 6))
    bars = sns.barplot(x=missing_pct.index, y=missing_pct.values, palette='Reds_r')
    
    for i, bar in enumerate(bars.patches):
        bars.annotate(f'{missing_pct.values[i]:.1f}%', 
                     (bar.get_x() + bar.get_width() / 2., bar.get_height()), 
                     ha='center', va='bottom', fontsize=10, 
                     xytext=(0, 5), textcoords='offset points')
                     
    plt.title('Tỷ lệ dữ liệu bị khuyết thiếu (Missing Values) theo từng trường', fontsize=15, fontweight='bold', pad=15)
    plt.xlabel('Tên trường (Cột)', fontsize=12)
    plt.ylabel('Tỷ lệ khuyết thiếu (%)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, max(missing_pct.max() * 1.1, 10)) # Chừa khoảng trống bên trên
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'advanced_missing_values.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   -> Đã lưu: advanced_missing_values.png")

def plot_scatter(df):
    """Vẽ biểu đồ Scatter Plot so sánh độ dài tiêu đề và nội dung"""
    print("5. Đang vẽ biểu đồ Scatter Plot (Title vs Content Length)...")
    
    # Tính số từ cho tiêu đề
    df['title_word_count'] = df['title'].fillna('').apply(lambda x: len(str(x).split()))
    # Đảm bảo đã có word_count của main_content
    if 'word_count' not in df.columns:
        df['word_count'] = df['main_content'].fillna('').apply(lambda x: len(str(x).split()))
        
    # Lọc bỏ các bài không có tiêu đề hoặc nội dung
    df_valid = df[(df['word_count'] > 0) & (df['title_word_count'] > 0)]
    
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df_valid, x='title_word_count', y='word_count', hue='domain', alpha=0.5, palette='Set1')
    
    plt.title('Scatter Plot: Mối quan hệ giữa Độ dài Tiêu đề và Độ dài Nội dung', fontsize=15, fontweight='bold', pad=15)
    plt.xlabel('Độ dài Tiêu đề (Số từ)', fontsize=12)
    plt.ylabel('Độ dài Nội dung (Số từ)', fontsize=12)
    
    # Cắt bỏ các Outliers quá xa để biểu đồ dễ nhìn hơn
    plt.xlim(0, df_valid['title_word_count'].quantile(0.99))
    plt.ylim(0, df_valid['word_count'].quantile(0.95))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'advanced_scatter_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   -> Đã lưu: advanced_scatter_plot.png")

def main():
    if not CSV_PATH.exists():
        print(f"Không tìm thấy file {CSV_PATH.name}. Vui lòng kiểm tra lại.")
        return
        
    df = load_data()
    
    if df.empty:
        print("File CSV rỗng.")
        return
        
    print(f"Tổng số bản ghi: {len(df):,}")
    
    all_tags = plot_top_tags(df)
    plot_wordcloud(all_tags)
    plot_content_length(df)
    plot_missing_values(df)
    plot_scatter(df)
    
    print("\nHoàn tất quá trình trực quan hóa!")

if __name__ == "__main__":
    main()

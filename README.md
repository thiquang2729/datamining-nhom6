# Nhom6 DataMining NEWS

Du an xay dung pipeline xu ly tin tuc cong nghe: tu du lieu tho, lam sach, chuan hoa, loai trung lap, tien xu ly NLP va tao dac trung TF-IDF de phuc vu cac buoc phan cum, gan nhan va huan luyen mo hinh.

## Cau truc thu muc

```text
data/
├── news_data.csv              # Du lieu tho, khong dua len GitHub neu file qua lon
├── cleaned_news.csv           # Du lieu sau lam sach/chuan hoa/loai trung
├── processed_news.csv         # Du lieu sau tien xu ly NLP, co cot processed_content
├── tfidf_features.pkl         # Ma tran dac trung TF-IDF
└── vietnamese-stopwords.txt   # Danh sach stopwords tieng Viet

src/
├── cleaning.py                # Lam sach du lieu rong, URL loi, do dai bat thuong
├── normalizer.py              # Chuan hoa Unicode, HTML, ky tu dac biet
├── deduplicator.py            # Loai trung lap bang MD5
├── preprocessing.py           # Tien xu ly NLP va trich xuat TF-IDF
├── pipeline.py                # Chay pipeline tong hop
├── modeling.py                # Phan cum / mo hinh
└── predict.py                 # Du doan du lieu moi

models/
└── vectorizer.pkl             # TF-IDF vectorizer da fit

notebooks/                     # Notebook va bieu do EDA/bao cao
demo/                          # Demo ung dung
config.json                    # Cau hinh duong dan va tham so pipeline
requirements.txt               # Thu vien can cai
```

## Quy trinh pipeline

```text
data/news_data.csv
    -> Buoc 1: Doc du lieu tho
    -> Buoc 2: Lam sach dong rong, URL loi, bai qua ngan/qua dai
    -> Buoc 3: Chuan hoa Unicode, xoa HTML, ky tu dac biet
    -> Buoc 4: Loai trung lap noi dung bang MD5
    -> data/cleaned_news.csv
    -> Buoc 5: Tien xu ly NLP va tao dac trung TF-IDF
    -> data/processed_news.csv
    -> data/tfidf_features.pkl + models/vectorizer.pkl
```

## Buoc tien xu ly NLP

File phu trach: `src/preprocessing.py`

Muc tieu cua buoc nay la bien noi dung bai viet da lam sach thanh dang van ban chuan cho may hoc, sau do chuyen van ban thanh vector so bang TF-IDF.

Quy trinh xu ly:

1. Doc cot noi dung bai viet tu `main_content`.
2. Chuyen chu ve dang thuong de thong nhat tu vung.
3. Tach tu don gian theo khoang trang va loai bo ky tu khong can thiet.
4. Loai bo stopwords tieng Viet tu `data/vietnamese-stopwords.txt`.
5. Tao cot moi `processed_content` trong DataFrame.
6. Fit `TfidfVectorizer` tren cot `processed_content`.
7. Luu ma tran TF-IDF vao `data/tfidf_features.pkl`.
8. Luu vectorizer da fit vao `models/vectorizer.pkl` de cac buoc sau dung lai.

Ket qua dau ra:

| File | Noi dung | Dung cho |
|---|---|---|
| `data/processed_news.csv` | Du lieu goc da them cot `processed_content` | Kiem tra, bao cao, gan nhan |
| `data/tfidf_features.pkl` | Ma tran dac trung TF-IDF cua cac bai viet | Phan cum, training model |
| `models/vectorizer.pkl` | Bo vectorizer da hoc tu tap du lieu | Bien doi du lieu moi khi predict |

Noi dung co the dua vao slide:

> Buoc tien xu ly NLP chuan hoa noi dung bai viet, loai bo stopwords tieng Viet va tao bieu dien vector TF-IDF. Ket qua la file `processed_news.csv` co cot `processed_content`, ma tran `tfidf_features.pkl` cho phan cum/mo hinh, va `vectorizer.pkl` de tai su dung khi du doan du lieu moi.

## Cach chay

```bash
pip install -r requirements.txt
python src/pipeline.py
```

Co the sua duong dan input/output va tham so tien xu ly trong `config.json`.

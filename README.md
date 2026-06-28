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

## Buoc tien xu ly NLP va TF-IDF

File phu trach: `src/preprocessing.py`

Muc tieu cua buoc nay la bien noi dung bai viet da lam sach thanh dang van ban chuan cho may hoc, sau do chuyen van ban thanh vector so bang TF-IDF. Day la buoc noi giua du lieu van ban va cac thuat toan phan cum/huan luyen mo hinh o phia sau.

### Quy trinh xu ly chi tiet

```text
cleaned_news.csv
    -> Doc cot main_content
    -> Chuyen ve chu thuong
    -> Tach tu tieng Viet bang underthesea.word_tokenize
    -> Loai stopwords trong data/vietnamese-stopwords.txt
    -> Tao cot processed_content
    -> Tao ma tran TF-IDF bang TfidfVectorizer
    -> Tuy chon giam chieu bang TruncatedSVD neu co cau hinh n_components
    -> Luu output cho cac buoc phan cum, gan nhan va huan luyen mo hinh
```

Giai thich tung buoc:

1. **Doc du lieu dau vao:** Lay du lieu da qua lam sach tu `data/cleaned_news.csv`.
2. **Lay cot noi dung:** Su dung cot `main_content` lam nguon van ban chinh.
3. **Chuyen chu thuong:** Dua tat ca van ban ve lowercase de cac tu giong nhau khong bi tinh thanh nhieu tu khac nhau.
4. **Tach tu tieng Viet:** Dung `underthesea.word_tokenize` de tach tu/cum tu tieng Viet hop ly hon so voi tach bang dau cach thong thuong.
5. **Loai stopwords:** Bo cac tu pho bien it mang y nghia phan loai, vi du nhu tu noi, tu dem, tu qua thong dung.
6. **Tao cot `processed_content`:** Luu van ban sau tien xu ly vao mot cot moi de de kiem tra va dung lai.
7. **Tao dac trung TF-IDF:** Dung `TfidfVectorizer` de bien moi bai viet thanh vector so. Tu nao quan trong hon trong bai viet se co trong so cao hon.
8. **Giam chieu neu can:** Neu cau hinh `n_components`, dung `TruncatedSVD` de giam so chieu, giup cac buoc mo hinh chay nhe hon.
9. **Luu ket qua:** Luu file du lieu da xu ly, ma tran TF-IDF va vectorizer da fit.

Ket qua dau ra:

| File | Noi dung | Dung cho |
|---|---|---|
| `data/processed_news.csv` | Du lieu goc da them cot `processed_content` | Kiem tra, bao cao, gan nhan |
| `data/tfidf_features.pkl` | Ma tran dac trung TF-IDF cua cac bai viet | Phan cum, training model |
| `models/vectorizer.pkl` | Bo vectorizer da hoc tu tap du lieu | Bien doi du lieu moi khi predict |

### Truc quan hoa buoc tien xu ly NLP

Sau khi tao `processed_content` va ma tran TF-IDF, co the xem cac bieu do trong thu muc `notebooks/`:

| Bieu do | Noi dung the hien | File |
|---|---|---|
| Phan phoi so tu truoc/sau xu ly | Cho thay van ban sau khi xu ly ngan gon hon, bot nhieu tu nhieu/tu dung | `notebooks/preprocessing_01_word_count_distribution.png` |
| So tu trung binh truoc/sau xu ly | So sanh trung binh moi bai viet truoc xu ly, sau xu ly va so tu da bi loai | `notebooks/preprocessing_02_word_reduction_bar.png` |
| Top tu/cum tu theo TF-IDF | Hien thi nhung tu/cum tu noi bat nhat trong tap tin tuc sau khi vector hoa | `notebooks/preprocessing_03_top_tfidf_terms.png` |

Nhan xet de dua vao slide:

> Sau tien xu ly, so tu trung binh moi bai giam tu khoang 703 tu xuong 415 tu, tuc la da loai bot khoang 289 tu nhieu/khong can thiet tren moi bai. Bieu do TF-IDF sau khi mo rong stopwords tap trung hon vao cac tu khoa noi dung cua tap tin tuc cong nghe, vi du `ai`, `iphone`, `cong_nghe`, `apple`, `viet_nam`, `phat_trien`, `du_lieu`, `ung_dung`.

Hinh anh minh hoa:

![Phan phoi so tu truoc/sau tien xu ly](notebooks/preprocessing_01_word_count_distribution.png)

![So tu trung binh truoc/sau tien xu ly](notebooks/preprocessing_02_word_reduction_bar.png)

![Top tu/cum tu noi bat theo TF-IDF](notebooks/preprocessing_03_top_tfidf_terms.png)

### Tom tat ngan de dua vao slide

Pipeline xu ly tin tuc di tu du lieu tho `news_data.csv`, sau do lam sach dong loi, chuan hoa van ban, loai bai trung lap va chay tien xu ly NLP. O buoc NLP, he thong doc cot `main_content`, chuyen chu thuong, tach tu tieng Viet bang `underthesea`, loai stopwords, tao cot `processed_content`, roi bien van ban thanh ma tran TF-IDF. Ket qua dung cho cac buoc sau gom `processed_news.csv`, `tfidf_features.pkl` va `models/vectorizer.pkl`.

```text
Du lieu tho
    -> Lam sach
    -> Chuan hoa
    -> Loai trung lap
    -> Tien xu ly NLP
    -> TF-IDF
    -> Phan cum / gan nhan / huan luyen mo hinh
```

### Vai tro cua cac file trong buoc NLP

| File | Vai tro |
|---|---|
| `src/preprocessing.py` | Chua class `TextPreprocessor` va ham `preprocess_nlp` de xu ly van ban, tao TF-IDF va luu output |
| `src/pipeline.py` | Goi `preprocess_nlp` trong buoc 5 cua pipeline tong |
| `config.json` | Cau hinh duong dan stopwords, cot dau vao, cot dau ra, noi luu TF-IDF va vectorizer |
| `data/vietnamese-stopwords.txt` | Danh sach tu dung tieng Viet can loai bo |
| `models/vectorizer.pkl` | Vectorizer da hoc tu tap du lieu, dung lai khi xu ly du lieu moi |

## Cach chay

```bash
pip install -r requirements.txt
python src/pipeline.py
```

Co the sua duong dan input/output va tham so tien xu ly trong `config.json`.

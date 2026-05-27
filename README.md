# Nhom8_DataMining_NEWS

```text
Nhom8_DataMining_NEWS/
|-- data/                 # Khong dua du lieu tho qua lon len GitHub
|-- notebooks/
|   |-- notebook.ipynb    # Noi chay thu nghiem, EDA, evaluation
|-- src/
|   |-- preprocessing.py  # Code lam sach, chuan hoa, rut trich dac trung
|   |-- modeling.py       # Code huan luyen mo hinh
|   |-- predict.py        # Code du doan cho du lieu moi
|-- demo/
|   |-- app.py            # Code giao dien web
|-- models/               # Noi luu model da train
|-- requirements.txt      # Danh sach thu vien can cai dat
|-- README.md             # File huong dan chay du an
```

## Phan Thanh vien 5 - Self-Training va Evaluation

Thanh vien 5 phu trach phan tu gan nhan du lieu chua co nhan, danh gia mo hinh va luu cac file dau ra.

### File phu trach

```text
src/modeling.py
models/
notebooks/notebook.ipynb
```

### Noi dung trong `src/modeling.py`

- Chia tap train/test bang Stratified Split.
- Huan luyen mo hinh Logistic Regression co so.
- Dung `CalibratedClassifierCV` de hieu chuan xac suat du doan.
- Du doan nhan cho du lieu chua co nhan.
- Chon cac mau co `confidence >= 0.9` de tao pseudo-label.
- Huan luyen lai model bang du lieu co nhan ban dau va pseudo-label.
- Danh gia model bang Accuracy, F1-score, Confusion Matrix va Classification Report.
- Luu model va ket qua vao thu muc `models/`.

### Du lieu can co de chay

Dat file du lieu vao thu muc `data/`:

```text
data/train_data.json
data/unlabeled_data.json
```

File train can co cot van ban va cot nhan.

Cot van ban co the dat ten:

```text
text, content, article_content, clean_text, description, title
```

Cot nhan co the dat ten:

```text
label, category, topic, subject
```

### Cach chay phan Thanh vien 5

Cai thu vien:

```bash
pip install -r requirements.txt
```

Chay pipeline Self-Training:

```bash
python src/modeling.py
```

Neu muon doi nguong confidence:

```bash
python src/modeling.py --threshold 0.9
```

### File ket qua sinh ra

Sau khi chay thanh cong, chuong trinh se tao:

```text
models/final_model_calibrated.pkl
models/pseudo_labels.json
models/evaluation_report.json
```

Luu y: file du lieu lon va model `.pkl` khong nen push len GitHub neu dung luong qua lon.

### Lenh Git de day phan Thanh vien 5

```bash
git status
git add src/modeling.py notebooks/notebook.ipynb models/.gitkeep .gitignore requirements.txt HUONG_DAN_THANH_VIEN_5.md README.md
git commit -m "Add member 5 self-training pipeline"
git push origin main
```

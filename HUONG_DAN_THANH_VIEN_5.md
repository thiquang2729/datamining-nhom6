# Huong dan phan viec Thanh vien 5

## Nhiem vu

Thanh vien 5 phu trach phan Self-Training, danh gia mo hinh va luu file dau ra:

- Tao vong lap gan nhan gia cho du lieu chua co nhan.
- Dung `CalibratedClassifierCV` de lay xac suat du doan on dinh hon.
- Chon cac bai bao co `confidence >= 0.9` lam pseudo-label.
- Huan luyen lai model bang du lieu co nhan ban dau + pseudo-label.
- Danh gia bang Accuracy, F1-score, Confusion Matrix.
- Luu `models/final_model_calibrated.pkl`.
- Luu `models/pseudo_labels.json`.
- Luu `models/evaluation_report.json`.

## File da lam

- `src/modeling.py`: code chinh cho training, self-training, evaluation va save output.
- `notebooks/notebook.ipynb`: notebook mau de chay va ve bieu do.
- `models/.gitkeep`: giu thu muc `models/` tren GitHub.
- `.gitignore`: tranh push data lon va model pkl len GitHub.
- `requirements.txt`: cac thu vien can cai.

## Cach chay

Dat du lieu vao:

```text
data/train_data.json
data/unlabeled_data.json
```

File train nen co cot van ban va cot nhan.

Cot van ban co the la mot trong cac ten:

```text
text, content, article_content, clean_text, description, title
```

Cot nhan co the la mot trong cac ten:

```text
label, category, topic, subject
```

Cai thu vien:

```bash
pip install -r requirements.txt
```

Chay pipeline:

```bash
python src/modeling.py
```

Neu muon doi nguong confidence:

```bash
python src/modeling.py --threshold 0.9
```

## File sinh ra sau khi chay

```text
models/final_model_calibrated.pkl
models/pseudo_labels.json
models/evaluation_report.json
```

Luu y: `.pkl` va du lieu lon dang duoc ignore de tranh lam nang repository. Neu giang vien yeu cau nop model, co the bo ignore rieng file do hoac nen nop qua Google Drive.

## Cach day len GitHub

Kiem tra file thay doi:

```bash
git status
```

Them file:

```bash
git add src/modeling.py notebooks/notebook.ipynb models/.gitkeep .gitignore requirements.txt HUONG_DAN_THANH_VIEN_5.md
```

Commit:

```bash
git commit -m "Add member 5 self-training pipeline"
```

Day len GitHub:

```bash
git push origin main
```

import pandas as pd

labeled = pd.read_csv("data/sample_for_labeling.csv")
predicted = pd.read_csv("data/sample_predicted.csv")

labeled["predicted_label"] = predicted["predicted_label"]

# Chỉ giữ dòng 2 bên đồng thuận
clean = labeled[labeled["label"] == labeled["predicted_label"]]

clean.to_csv("data/sample_for_labeling_clean.csv", index=False, encoding="utf-8-sig")
print(f"Còn lại: {len(clean)}/1000 dòng")
print(clean["label"].value_counts())
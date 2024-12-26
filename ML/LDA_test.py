import numpy as np
import csv
from joblib import load
from sklearn.metrics import accuracy_score

# 測試集資料路徑
test_file_path = r"D:\大學課程\四上\電腦視覺\期末\test_dataset.txt"

# 取測試資料
data, img_names = [], []
with open(file=test_file_path, mode="r", encoding="utf-8") as file:
    for line in file:
        line = line.strip().split()
        features, label = [float(x) for x in line[:-2]], line[-2]  # 特徵轉為浮點數、標籤
        data.append(features + [label])
        img_names.append(line[-1])

# 資料轉型
data = np.array(object=data, dtype=object)
x = data[:, :-1].astype(dtype=np.float32)  # 特徵
y = data[:, -1]   # 標籤

# 預測結果
model = load("LDA_model.joblib")    # 載入模型
y_pred = model.predict(x)   # 預測
accuracy = accuracy_score(y_true=y, y_pred=y_pred)  # 計算準確度

# 寫檔紀錄
with open(file="LDA_result.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["image_names", "real class", "prediction class", accuracy])  # 標題 + 準確度
    writer.writerows(zip(img_names, y, y_pred))  # 內容
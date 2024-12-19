import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from joblib import dump

# 訓練集資料路徑
train_file_path = r"D:\大學課程\四上\電腦視覺\期末\train_dataset.txt"

# 取訓練資料
data = []
with open(file=train_file_path, mode="r", encoding="utf-8") as file:
    for line in file:
        line = line.strip().split()
        features, label = [float(x) for x in line[:-2]], line[-2]  # 特徵轉為浮點數、標籤
        data.append(features + [label])

# 資料轉型
data = np.array(object=data, dtype=object)
x = data[:, :-1].astype(dtype=np.float32)  # 特徵
y = data[:, -1]   # 標籤

# 訓練集、驗證集
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

# 訓練模型
model = LDA()
model.fit(X=x_train, y=y_train)

# 儲存模型
dump(value=model, filename="LDA_model.joblib")

# 計算學習曲線
train_sizes, train_scores, val_scores = learning_curve(
    estimator=model,
    X=x_train,
    y=y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),  # 訓練集大小
    cv=5,   # 交叉驗證
    scoring="accuracy",
    n_jobs=-1
)

# 計算每個訓練集的平均分數和標準差
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

# 畫學習曲線
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label="training score", color="blue")
plt.plot(train_sizes, val_mean, label="cross-validation score", color="red")
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="blue", alpha=0.2)
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, color="red", alpha=0.2)
plt.title("learning curve for LDA", fontsize=14)
plt.xlabel("training size", fontsize=12)
plt.ylabel("accuracy", fontsize=12)
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.savefig("LDA_learning_curve.jpg")
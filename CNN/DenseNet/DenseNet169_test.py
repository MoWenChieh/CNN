import tensorflow as tf
import numpy as np
import csv
from pathlib import Path

# 設定測試圖片
test_root = r'D:\大學課程\四上\電腦視覺\期末\test_dataset'  # 測試資料集的資料夾
image_paths = [str(image) for image in Path(test_root).rglob("*.jpg")]  # 測試圖片的路徑
real_class_names = [image_path.parent.name for image_path in Path(test_root).rglob("*.jpg")] # 測試圖片的類別名稱 (答案)

# 設定模型
model = tf.keras.models.load_model('DenseNet169_model.h5')    # 載入模型
image_size = model.input_shape  # 圖片大小
batch_size = 64  # 批次大小

# 測試資料
test_data = tf.keras.preprocessing.image_dataset_from_directory(
    directory=test_root,
    image_size=(image_size[1], image_size[2]),
    batch_size=batch_size,
    label_mode='categorical',
    shuffle=False  # 禁用打亂讀取順序
)

# 圖片預處理


# 預測資料
predictions = model.predict(test_data)  # 預測每張圖片的類別機率分布
prediction_class_names = np.argmax(predictions, axis=1)  # 決定圖片類別 (數字)
class_no = sorted(list(set(real_class_names)))   # 類別順序
prediction_class_names = [class_no[i] for i in prediction_class_names]  # 數字轉為類別名稱 (字串)

# 計算準確度
accuracy = sum([i == j for i, j in zip(real_class_names, prediction_class_names)]) / len(real_class_names)

# 寫檔紀錄
image_names = [Path(image_path).name for image_path in image_paths] # 圖片的原始檔名
rows = zip(image_names, real_class_names, prediction_class_names)    # 資料
with open('DenseNet169_result.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['image_names', 'real class', 'prediction class', accuracy])  # 標題 + 準確度
    writer.writerows(rows)  # 內容
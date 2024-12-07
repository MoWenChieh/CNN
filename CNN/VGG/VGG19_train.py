import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

# 設定
train_root = r'D:\大學課程\四上\電腦視覺\期末\train_dataset'  # 訓練資料集的資料夾
num_classes = len([file for file in os.listdir(train_root) if os.path.isdir(os.path.join(train_root, file))])   # 類別數量
image_size = (224, 224)  # 圖片大小
batch_size = 16  # 批次大小
epochs = 50  # 訓練次數

# 訓練資料
tdata = tf.keras.preprocessing.image_dataset_from_directory(
    directory=train_root,
    validation_split=0.2,
    subset="training",
    seed=7414,  # 隨機
    image_size=image_size,
    batch_size=batch_size,
    label_mode='categorical'
)

# 驗證資料
vdata = tf.keras.preprocessing.image_dataset_from_directory(
    directory=train_root,
    validation_split=0.2,
    subset="validation",
    seed=7414,  # 隨機
    image_size=image_size,
    batch_size=batch_size,
    label_mode='categorical'
)

# 預處理資料
tdata = tdata.prefetch(buffer_size=batch_size)
vdata = vdata.prefetch(buffer_size=batch_size)
input_shape = (image_size[0], image_size[1], 3)  # 圖片輸入的形狀

# VGG19 預訓練
feature_model = tf.keras.applications.VGG19(
    include_top=False,  # 無頂層分類層
    weights='imagenet',  # 預層訓練權重
    input_shape=input_shape,
    pooling='max',   # 最大池化
    classes=num_classes, # 類別數量
    classifier_activation='softmax' # 最終分類層的激活函數
)

# 建立 VGG19 模型
model = models.Sequential([
    feature_model,
    layers.Flatten(),  # Flatten層將輸出攤平
    layers.Dense(4096, activation='relu'),  # 全連接層
    layers.BatchNormalization(),  # BatchNormalization層 (自動調整每層的輸入分佈，防止梯度消失問題，並減少訓練過程中的波動)
    layers.Dropout(0.5),  # Dropout層防止過擬合
    layers.Dense(4096, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(4096, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')  # 輸出層
])
model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])    # 編譯模型
history = model.fit(x=tdata, epochs=epochs, validation_data=vdata, batch_size=batch_size)  # 訓練模型
model.save('VGG19_model.h5')    # 保存模型

# 宣告畫布
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# 繪製損失曲線
ax[0].plot(history.history['loss'], label='Train Loss')
ax[0].plot(history.history['val_loss'], label='Validation Loss')
ax[0].set_title('VGG19 Loss')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Loss')
ax[0].legend(loc='upper left')

# 繪製學習曲線
ax[1].plot(history.history['accuracy'], label='Train Accuracy')
ax[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
ax[1].set_title('VGG19 Accuracy')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Accuracy')
ax[1].legend(loc='upper left')

# 保存學習曲線圖片
plt.tight_layout()
plt.savefig("VGG19_learning_curve.jpg")
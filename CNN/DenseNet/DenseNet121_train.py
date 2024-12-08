import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

# 設定
train_root = r'D:\大學課程\四上\電腦視覺\期末\train_dataset'  # 訓練資料集的資料夾
image_size = (224, 224)  # 圖片大小
batch_size = 64  # 批次大小
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

# 圖片預處理


# 載入官方模型
feature_model = tf.keras.applications.DenseNet121(
    include_top=False,  # 無全連接層
    weights='imagenet',  # 全連接層訓練權重
    input_shape=(image_size[0], image_size[1], 3),  # 圖片輸入的形狀
    pooling='max'   # 最大池化
)
feature_model.trainable = False # 凍結權重

# 建立模型
model = models.Sequential([
    feature_model,
    layers.Flatten(),  # 攤平向量
    layers.Dense(256, activation='relu'),  # 全連接層
    layers.BatchNormalization(),  # 調整輸入分佈
    layers.Dropout(0.5),  # 阻止過擬合
    layers.Dense(len(tdata.class_names), activation='softmax')  # 最終分類層
])

model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])    # 編譯模型
history = model.fit(x=tdata, epochs=epochs, validation_data=vdata, batch_size=batch_size)  # 訓練模型
model.save('DenseNet121_model.h5')    # 保存模型

# 宣告畫布
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# 繪製損失曲線
ax[0].plot(history.history['loss'], label='Train Loss')
ax[0].plot(history.history['val_loss'], label='Validation Loss')
ax[0].set_title('DenseNet121 Loss')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Loss')
ax[0].legend(loc='upper left')

# 繪製學習曲線
ax[1].plot(history.history['accuracy'], label='Train Accuracy')
ax[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
ax[1].set_title('DenseNet121 Accuracy')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Accuracy')
ax[1].legend(loc='upper left')

# 保存學習曲線圖片
plt.tight_layout()
plt.savefig("DenseNet121_learning_curve.jpg")
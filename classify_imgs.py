import os, shutil

def create_folders(folders):
    for folder in folders:
        if os.path.exists(folder):  # 判斷資料夾是否存在
            shutil.rmtree(folder)   # 刪除資料夾
        os.makedirs(name=folder, exist_ok=True)  # 建立資料夾

def classify_imgs(folder, path):
    class_name = os.path.basename(os.path.dirname(path))   # 取得圖片的類別名稱
    class_folder = os.path.join(folder, class_name)
    os.makedirs(name=class_folder, exist_ok=True)   # 建立類別資料夾
    shutil.copy(src=path, dst=class_folder)     # 複製到指定資料夾

img_paths = []  # 儲存圖片路徑
for dirpath, dirnames, filenames in os.walk(r"D:\大學課程\四上\電腦視覺\期中\pic"):  # 取出所有圖片的路徑
    for filename in filenames:
        if filename.lower().endswith(".jpg"):   # 圖片
            img_paths.append(os.path.join(dirpath, filename))

test_no = []    # 儲存測試圖片的編號
with open(file=os.path.join(r"D:\大學課程\四上\電腦視覺\期末\final report - 1", "query.txt"), mode="r", encoding="utf-8") as file:
    for line in file:
        test_no.append(int(line.strip()))    # 編號

test_paths = [img_paths[i-1] for i in test_no]  # 測試集圖片路徑
train_paths = set(img_paths) - set(test_paths)  # 訓練集圖片路徑

test_folder, train_folder = "test_dataset", "train_dataset"
create_folders(folders=[test_folder, train_folder]) # 建立存放測試與訓練集的圖片資料夾
for test_path in test_paths:
    classify_imgs(test_folder, test_path)
for train_path in train_paths:
    classify_imgs(train_folder, train_path)
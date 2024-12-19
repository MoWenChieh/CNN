import os

folder = r"D:\大學課程\四上\電腦視覺\期末\final report - 1"
txts = ["fnormal.txt", "name.txt", "query.txt"]
test_txt, train_txt = "test_dataset.txt", "train_dataset.txt"

# 先取出所有的特徵和標籤，行數當字典索引
features_and_labels = {}
with open(file=os.path.join(folder, txts[0]), mode="r", encoding="utf-8") as file1, \
     open(file=os.path.join(folder, txts[1]), mode="r", encoding="utf-8") as file2:
    row = 1
    for line1, line2 in zip(file1, file2):
        feature = line1.strip().split() # 特徵
        label = line2.strip().split()[::-1]    # 標籤 (圖片名稱順便寫，反轉方便寫檔)
        features_and_labels[row] = feature + label
        row += 1

# 取出測試集的資料編號
no = []
with open(file=os.path.join(folder, txts[2]), mode="r", encoding="utf-8") as file:
    for line in file:
        no.append(int(line.strip()))
no.sort()   # 排序，方便寫檔

# 寫檔 (測試集、訓練集)
with open(file=os.path.join(os.path.dirname(os.path.abspath(__file__)), test_txt), mode="w", encoding="utf-8") as file1, \
     open(file=os.path.join(os.path.dirname(os.path.abspath(__file__)), train_txt), mode="w", encoding="utf-8") as file2:
    for index in features_and_labels.keys():
        if index in no:
            file1.write(" ".join(features_and_labels[index]) + "\n")
        else:
            file2.write(" ".join(features_and_labels[index]) + "\n")
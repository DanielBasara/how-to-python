# %%
# 数据处理测试
# 这是一个为了弄懂广岛水处理厂水池异常检查AI程序和对少量数据或模型进行前处理和训练的python程序

# %%
# json文件的编辑，保存和读取

# %%
from multiprocessing import Process, Queue
from multiprocessing.queues import Queue
import queue
import matplotlib.pyplot as plt
from tensorflow.python.ops.gen_data_flow_ops import tensor_array
from tqdm import tqdm
import time
import os
import json
import pandas as pd
import IPython.display as display
from pathlib import Path
import numpy as np
from PIL import Image
from datetime import datetime
import cv2
import seaborn as sns
import tensorflow as tf
# %%
file_path = 'json/test.json'


new_dirt = {'family': {'number': 3, 'member': {
    'father': 'jack', 'mother': 'lili', 'son': 'Daniel'}}}

# 写入jason文件
with open(file_path, "w") as write_f:
    json.dump(new_dirt, write_f)

# 把文件打开，并把字符串变换为数据类型
with open(file_path, 'r') as load_f:
    load_dirt = json.load(load_f)
print(load_dirt['family']['member'].get('father'))

# %%

folder_path = 'json/create_filelist/'

file_name = os.listdir(folder_path)

with open(folder_path+file_name[0], 'r', encoding='UTF-8') as f:
    args = json.load(f)
# %%进度条
# iterable: 可迭代的对象, 在手动更新时不需要进行设置
# desc: 字符串, 左边进度条描述文字
# total: 总的项目数
# leave: bool值, 迭代完成后是否保留进度条
# file: 输出指向位置, 默认是终端, 一般不需要设置
# ncols: 调整进度条宽度, 默认是根据环境自动调节长度, 如果设置为0, 就没有进度条, 只有输出的信息
# unit: 描述处理项目的文字, 默认是'it', 例如: 100 it/s, 处理照片的话设置为'img' ,则为 100 img/s
# unit_scale: 自动根据国际标准进行项目处理速度单位的换算, 例如 100000 it/s >> 100k it/s


def tic():
    time.sleep(0.5)


with tqdm(total=100000, desc='test', leave=True, ncols=100, unit='B', unit_scale=True) as pbar:
    for i in range(10):
        tic()
        pbar.update(10000)

# %%
# path用于路径的获取使用，可以方便在不同的位置运行程序也依然可以找到正确的路径
Path.cwd()  # 获取当前路径
Path.cwd().parent.parent  # 获取上上层目录

Path.cwd().parent.joinpath(*paths)  # 拼接路径
Path('data/test').mkdir(parents=True, exist_ok=True)  # 创建目录
Path('json/test.json').rename('')

# %% テーブルの項目を作る
project_directory = "json/project.json"
img_list_path = 'database/database.csv'


with open(project_directory, "r", encoding='UTF-8') as f:
    data = json.load(f)
    img_raw_path = data.get("生データ保存パス")

# %% 生データのファイル名全部読み取る ****
filePath_list = []
timestamp_list = []
file_path = Path(
    r"\\10.101.65.115\広島西部\002_広島西部水資源再生センター\02_data\05_静止画\07_東3-2系終沈\2018.12.17 回収\00001057")
for i, j, k in os.walk(file_path):
    for filename in k:
        if os.path.splitext(filename)[1] == ".jpg":
            x = os.path.join(i, filename)
            filePath_list.append(x)
            timestamp_list.append(datetime.fromtimestamp(os.path.getmtime(x)))

file_list = {"datatime": timestamp_list, "filename": filePath_list}
file_list_DF = pd.DataFrame(file_list)
file_list_DF.to_csv("database/file_list.csv", encoding="utf_8_sig")

# %% check img
file_list_DF = pd.read_csv("database/file_list.csv", index_col=0)
check_num = 501
check_img_date = file_list_DF.iloc[check_num][0]
check_img_path = file_list_DF.iloc[check_num][1]
check_img = Image.open(check_img_path)
img = np.asarray(check_img)
plt.figure(figsize=(10, 10))
plt.imshow(img)
plt.show()
print(check_img_date)

# %% img resize
with open("json/img_resize.json", "r") as f:
    img_resize = json.load(f)
y0 = img_resize["crop_lat_n"]
y1 = img_resize["crop_lat_s"]
x0 = img_resize["crop_lon_w"]
x1 = img_resize["crop_lon_e"]
height = img_resize["resize_height"]
width = img_resize["resize_width"]
cropped_img = img[y0: y1, x0: x1]
cropped_img = cv2.resize(cropped_img, (height, width))

plt.figure(figsize=(10, 10))
plt.imshow(cropped_img)
plt.show()

# %%  HSV変換
hsv_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2HSV)
image_v = hsv_img[:, :, 2]
illum = image_v.mean()
plt.imshow(hsv_img, cmap="hsv")
plt.show()

cv2.imshow("hsv", hsv_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %% to gray
gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
x = np.array(gray_img).reshape(256, 256, 1)
plt.imshow(x, cmap="gray")
plt.show()

# %% to tensor
a = tf.convert_to_tensor(x)
b = a
c = tf.stack([a, b], axis=0)

# %% すべての画像をHSV変換で強化する
(img_num, _) = file_list_DF.shape
illum_list = []
for i in range(img_num):
    img = Image.open(file_list_DF.iloc[i][1])
    img = np.asarray(img)
    cropped_img = img[y0: y1, x0: x1]
    cropped_img = cv2.resize(cropped_img, (height, width))
    hsv_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2HSV)
    image_v = hsv_img[:, :, 2]
    illum_list.append(image_v.mean())
illum = {"datatime": file_list_DF["datatime"].values, "illum_mean": illum_list}
pd.DataFrame(illum).to_csv("database/illum_mean.csv")

# %%データ分析
illum_mean_list = pd.read_csv("database/illum_mean.csv")

x = range(150)
plt.bar(x, illum_mean_list["illum_mean"].values, label='graph 1')

plt.show()


fig = plt.figure()
plt.figure(figsize=(20, 10))
with plt.style.context('seaborn-poster') as st:
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(0, 120)
    sns.distplot(illum_mean_list["illum_mean"].values, label="illum", kde=False,
                 rug=False, bins=40, hist_kws={"alpha": 0.8}, ax=ax)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left",
              borderaxespad=0., frameon=True, edgecolor="blue")
plt.show()


# %% loop データセット１０００枚作る
file_list = pd.read_csv("database/file_list.csv", index_col=0)  # load csv

with open("json/img_resize.json", "r") as f:  # load resize data_json
    img_resize = json.load(f)
y0 = img_resize["crop_lat_n"]
y1 = img_resize["crop_lat_s"]
x0 = img_resize["crop_lon_w"]
x1 = img_resize["crop_lon_e"]
height = img_resize["resize_height"]
width = img_resize["resize_width"]

catch_num = 0
data_list_path = []
array_list = []


def resize_tensor(array, height, width):
    array = tf.reshape(array, [height, width, 1])
    return array


n = len(file_list)
for catch_num in range(len(file_list)):
    check_img_date = file_list.iloc[catch_num][0]
    check_img_path = file_list.iloc[catch_num][1]
    date = datetime.fromisoformat(check_img_date).strftime("%Y%m%d%H%M%S")
    save_path = Path("data/img", date+".jpg")
    data_list_path.append(save_path)
    check_img = Image.open(check_img_path)
    img = np.asarray(check_img)
    cropped_img = img[y0: y1, x0: x1]
    cropped_img = cv2.resize(cropped_img, (height, width))
    gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    tensor = resize_tensor(gray_img, height, width)
    array_list.append(tensor)
    cv2.imwrite(str(save_path), gray_img)
data_list = pd.DataFrame(file_list["datatime"])
data_list["data_path"] = data_list_path
data_list.to_csv("database/data_list.csv", encoding="utf_8_sig")
tensor_list = tf.stack(array_list, axis=0)
np.save("100.npy", tensor_list)

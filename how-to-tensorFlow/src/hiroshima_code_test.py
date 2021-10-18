# %%
# 数据处理测试
# 这是一个为了弄懂广岛水处理厂水池异常检查AI程序和对少量数据或模型进行前处理和训练的python程序

# %%
# json文件的编辑，保存和读取

# %%
from tqdm import tqdm
import time
import os
import json
import pandas as pd
import IPython.display as display
from pathlib import Path
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

paths = ['test', 'test.txt']
Path.cwd().parent.joinpath(*paths)  # 拼接路径
Path('data/test').mkdir(parents=True, exist_ok=True)  # 创建目录
Path('json/test.json').rename('')

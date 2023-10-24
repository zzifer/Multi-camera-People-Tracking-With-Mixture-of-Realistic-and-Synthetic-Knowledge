import gc
import os
import pickle

import numpy as np
from tqdm import tqdm

# 此文件用于处理行人检测的结果和特征数据，将它们整理并保存为Numpy数组或pickle文件

feat_list = []

FEATURE_DIR = "/mnt/Data/dataset/ReiD/AIC23_Track1_MTMC_Tracking/outputs/Features/trans_feat_yolo"
DETECTION_DIR = "/mnt/Data/CVIP-Lab-Work/Multi-Camera-People-Tracking/datasets/detections/Yolo_pretrain"

# 用于处理单个摄像头（scene_cam）的特征和检测结果
def process_cam(scene_cam):
    result = []
    # 加载特征数据文件，feats 是一个包含特征数据的字典
    feats = pickle.load(open(f"{FEATURE_DIR}/{scene_cam}_feature.pkl", "rb"))
    print("Load feat done")
    f = open(f"{DETECTION_DIR}/{scene_cam}.txt", "r")
    f = f.readlines()
    # 按照每行的第一个元素（帧号）进行排序
    f = sorted(f, key=lambda x: int(x.split(",")[0]))
    # print(f[0])
    for (idx, line) in  tqdm(enumerate(f),total=len(f)):
        # 将当前行按逗号分割，以获取各个字段的数据
        f = line.split(",")
        # 提取帧号、坐标、置信度等信息，并创建一个特征名feat_name
        fid = f[0].zfill(6)
        x = int(f[2])
        y = int(f[3])
        w = int(f[4])
        h = int(f[5])
        conf = float(f[6])

        feat_name = f"{scene_cam}_{fid}_{x}_{y}_{w}_{h}"
        feat_value = feats.pop(feat_name)
        # 创建一个包含帧号、坐标、置信度和特征值的列表_line
        _line = [fid, -1, x, y, w, h, conf, -1, -1, -1]
        _line.extend(feat_value.tolist())
        # 创建一个包含帧号、坐标、置信度和特征值的列表 _line
        result.append(_line)
        
    return result

# 遍历检测结果目录中的文件
for scene in os.listdir(DETECTION_DIR):
    # 移除文件扩展名（".txt"）
    scene = scene.replace(".txt", "")
    result = process_cam(scene)
    # 将处理后的结果转换为Numpy数组
    det_feat_npy = np.array(result)
    # np.save('{}.npy'.format(det_feat_npy), det_feat_npy)
    with open(f"{scene}.pkl", "wb") as f:
        # 使用 pickle 将数组保存为 .pkl 格式的文件
        pickle.dump(det_feat_npy, f)
    del result, det_feat_npy
    gc.collect()

# print(result[0])

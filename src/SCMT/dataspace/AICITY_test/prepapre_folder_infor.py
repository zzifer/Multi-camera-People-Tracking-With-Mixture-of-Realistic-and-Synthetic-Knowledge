import os
import cv2 
import numpy as np
from src.utils.utils import load_defaults
cfg = load_defaults(['configs/baseline.yaml'])
print(cfg['DATASETS']['ROOT_DIR'])
ROOT_DIR = os.path.join(cfg['DATASETS']['ROOT_DIR'], 'test')
print(ROOT_DIR)
# ROOT_DIR = "/mnt/ssd8tb/quang/AIC23_Track1_MTMC_Tracking/test"

# 为一个多摄像头的跟踪数据集生成相应的seqinfo.ini配置文件


cam_info = ""
# 设置帧率为30帧/秒
frame_rate=30
# 初始化length变量，用于存储每个摄像头的帧数
length = 0
# 图像的宽度和高度
width=1920
height=1080
ext = ".jpg"
# 构建一个包含序列信息的字符串，包括摄像头名称、图像文件夹、帧率、序列长度、图像宽度、图像高度和图像文件扩展名
infor = f"[Sequence]\nname={cam_info}\nimgDir=img1\nframeRate={frame_rate}\nSeqLength={length}\nimWidth={width}\nimHeight={height}\nimgExt={ext}\n"
scene_infor = dict()

# 遍历根目录下的所有文件和子目录
for scene in os.listdir(ROOT_DIR):
    # 如果文件名包含两个部分（通过'.'分隔），则跳过此迭代
    if len(scene.split('.')) == 2: continue
    # 在scene_infor字典中为当前场景创建一个空列表，用于存储摄像头信息 
    scene_infor[scene] = []
    # 遍历场景目录下的所有文件和子目录
    for cam in os.listdir(os.path.join(ROOT_DIR, scene)):
        if cam == 'map.png': continue
        if len(cam.split('.')) == 2: continue
        # 构建摄像头目录的完整路径
        cam_dir = os.path.join(ROOT_DIR, scene, cam)
        # 获取图像文件夹中的图像文件数量，然后更新length变量
        length = len(os.listdir(cam_dir +'/img1'))
        infor = f"[Sequence]\nname={cam}\nimgDir=img1\nframeRate={frame_rate}\nSeqLength={length}\nimWidth={width}\nimHeight={height}\nimgExt={ext}\n"

        # 使用with上下文管理器，打开当前摄像头目录下的seqinfo.ini文件，并将序列信息写入文件中
        with open(f"{cam_dir}/seqinfo.ini", "w") as f:
            f.write(infor)
        # 将摄像头名称添加到当前场景的摄像头列表中
        scene_infor[scene].append(cam)

# 创建一个全白图像（3通道），尺寸为1920x1080像素
img = np.ones((1080, 1920, 3), np.uint8) * 255.0
# 将图像保存为文件roi.png
cv2.imwrite("src/SCMT/track_roi/roi.png", img)
# print(scene_infor)

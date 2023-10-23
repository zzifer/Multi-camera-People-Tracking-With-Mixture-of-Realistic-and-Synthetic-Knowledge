import os
import json
import numpy as np
import PIL.Image as Image
from src.utils.opt import *
import src.utils.utils as util
import cv2
import os
from multiprocessing import Pool
from sys import stdout
import numpy as np

# 加载配置文件，将配置信息存储在cfg变量中
cfg = util.load_defaults()
# 从配置文件中获取数据集的根目录路径
ROOT_DATA_DIR = cfg["DATASETS"]["ROOT_DIR"]
print(ROOT_DATA_DIR)

fprint, endl = stdout.write, "\n"


ROLES = ["train", "validation", "test"]
IMAGE_FORMAT = ".jpg"  # ".png"


# 将视频文件转换成图像序列
# 函数接受一个包含四个元素的参数parameter_set，依次表示角色、场景、摄像机和摄像机目录
def video2image(parameter_set):
    role, scenario, camera, camera_dir = parameter_set
    fprint(f"[Processing] {role} {scenario} {camera}{endl}")
    # 定义了保存图像序列的目录，这个目录是根据摄像机目录构建的
    imgs_dir = f"{camera_dir}/img1"
    if not os.path.exists(imgs_dir):
        os.makedirs(imgs_dir)
    # 通过cv2.VideoCapture打开视频文件，然后使用一个循环来逐帧读取视频帧，并将每一帧保存为图像文件
    cap = cv2.VideoCapture(f"{camera_dir}/video.mp4")
    current_frame = 1
    ret, frame = cap.read()
    while ret:
        frame_file_name = f"{str(current_frame).zfill(6)}{IMAGE_FORMAT}"
        cv2.imwrite(f"{imgs_dir}/{frame_file_name}", frame)
        ret, frame = cap.read()
        current_frame += 1
    fprint(f"[Done] {role} {scenario} {camera}{endl}")


def main():
    # 使用嵌套循环遍历不同的角色、场景和摄像机，将参数集合添加到parameter_sets中
    parameter_sets = []
    for each_role in ROLES:
        role_dir = f"{ROOT_DATA_DIR}/{each_role}"
        scenarios = os.listdir(role_dir)
        for each_scenario in scenarios:
            scene = each_scenario
            scenario_dir = f"{role_dir}/{each_scenario}"
            cameras = os.listdir(scenario_dir)
            # cameras = ["c019"]
            for each_camera in cameras:
                cam = each_camera
                if "map" in each_camera:
                    continue
                camera_dir = f"{scenario_dir}/{each_camera}"                
                parameter_sets.append(
                    [each_role, each_scenario, each_camera, camera_dir]
                )

    # 使用multiprocessing.Pool来创建一个包含15个进程的进程池，然后使用pool.map来并行地调用video2image函数来处理参数集合
    pool = Pool(processes=15)
    pool.map(video2image, parameter_sets)
    pool.close()


if __name__ == "__main__":
    main()

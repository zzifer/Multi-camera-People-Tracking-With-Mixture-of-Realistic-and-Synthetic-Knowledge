import os
TARGET_DIR = '/mnt/ssd8tb/quang/detection/test/images'

from sys import stdout
from pathlib import Path

def main():
    # 使用os模块的listdir方法列出TARGET_DIR目录下的所有文件和目录名，并将这些名字存储在变量images中
    images = os.listdir(TARGET_DIR)
    # 打开（或创建）一个名为track1_synthetic.txt的文件用于写入，存储与合成图像相关的路径
    f_synthetic = open("./track1_synthetic.txt", "w")
    # 打开（或创建）一个名为track1_S001.txt的文件用于写入，存储与真实图像相关的路径
    f_real = open("./track1_S001.txt", "w")
    for image in images:
        IMAGE_PATH = os.path.join(TARGET_DIR, image)
        if 'S001' in image:
            f_real.write(IMAGE_PATH)
            f_real.write("\n")
        else:
            f_synthetic.write(IMAGE_PATH)
            f_synthetic.write("\n")
if __name__ == "__main__":
    main()

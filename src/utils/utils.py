import cv2 as cv, cv2
import os
from typing import List
import torch
from pathlib import Path
from dotenv import load_dotenv
import os
import numpy as np
import random
import platform
from .opt import Opts, Config


"""
显示图像窗口，并可以指定窗口的位置。它接受窗口名称 winname、图像 img，以及窗口的 x 和 y 坐标位置。
函数首先使用cv.namedWindow创建一个具有指定名称的窗口，然后使用cv.moveWindow将窗口移动到指定的位置 (x, y)，最后使用cv.imshow显示图像在窗口中
"""
def showInMovedWindow(winname, img, x, y):
    cv.namedWindow(winname)  # Create a named window
    cv.moveWindow(winname, x, y)  # Move it to (x,y)
    cv.imshow(winname, img)


"""
getCamCapture 函数用于获取视频捕捉对象。它接受一个参数 data，该参数可以是一个视频文件路径或一个目录。
如果 data 是一个目录，函数会假定该目录包含图像序列，并使用cv.VideoCapture创建一个视频捕捉对象，
将其初始化为读取指定目录中的图像序列。函数还返回总帧数（如果是目录的话）
"""
def getCamCapture(data):
    """Returns the camera capture from parsing or a pre-existing video.

    Args:
      isParse: A boolean value denoting whether to parse or not.

    Returns:
      A video capture object to collect video sequences.
      Total video frames

    """
    total_frames = None
    if os.path.isdir(data):
        cap = cv.VideoCapture(data + "/input/in%06d.jpg")
        total_frames = len(os.listdir(os.path.join(data, "input")))
    else:
        cap = cv.VideoCapture(data)
    return cap, total_frames


"""
它接受一个主字典 mapping 和多个要更新的字典 updating_mappings，
然后递归地将这些更新合并到主字典中。这个函数可以用于配置的深度更新操作
"""
def deep_update(mapping: dict, *updating_mappings: dict()) -> dict():
    updated_mapping = mapping.copy()
    for updating_mapping in updating_mappings:
        for k, v in updating_mapping.items():
            if (
                k in updated_mapping
                and isinstance(updated_mapping[k], dict)
                and isinstance(v, dict)
            ):
                updated_mapping[k] = deep_update(updated_mapping[k], v)
            else:
                updated_mapping[k] = v
    return updated_mapping


def get_device(choose_device):
    if torch.cuda.is_available():
        device = "cuda:0"
    if choose_device == "cpu" or device == "cpu":
        return "cpu"
    return device


"""
从嵌套字典中提取所有键。
它接受一个嵌套字典 _dict，并返回一个包含所有键的列表。该函数递归遍历嵌套字典，并将每个键添加到结果列表中
"""
def get_dict_infor(_dict: dict) -> List[str]:
    res_list = []
    for k, v in _dict.items():
        if isinstance(v, dict):
            get_list = get_dict_infor(v)
            for val in get_list:
                res_list.append(str(k) + "." + str(val))
        else:
            res_list.append(str(k))
    return res_list


"""
根据配置列表 cfg 和值 value 创建一个嵌套字典。
如果配置列表长度为1，它将返回一个包含一个键和值的字典，否则递归调用以创建嵌套字典
"""
def update_cfg(cfg: List, value):
    if len(cfg) == 1:
        return {cfg[0]: value}
    return {cfg[0]: update_cfg(cfg[1:], value)}


"""
环境变量文件加载配置。
它首先使用load_dotenv加载.env文件中的环境变量。
然后，它获取配置字典中所有的键，检查每个键对应的环境变量是否存在，如果存在，将其值更新到配置字典中，最后返回更新后的配置
"""
def load_enviroment_path(cfg: dict):
    load_dotenv(Path(".env"))

    variables = get_dict_infor(cfg)

    for var in variables:
        variable_value = os.getenv(var)
        if variable_value is None:
            continue
        params = var.split(".")
        temp = update_cfg(params, variable_value)
        cfg = deep_update(cfg, temp)

    return cfg


"""
设置随机种子，以确保可重复的随机性行为。
它接受一个种子值 seed，然后使用torch、np.random和random库设置随机种子，以及调整CUDA的行为
"""
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_device_name():
    return platform.node()


"""
用于将字典转换为实例，使得可以使用点标记（属性）访问字典键。
它接受一个字典 config，并返回一个特殊类型的实例，该实例允许通过属性名访问字典的键。这个函数的目的是让配置信息更易于访问和操作
"""
def config2object(config):
    """
    Convert dictionary into instance allowing access to dictionary keys using
    dot notation (attributes).
    """

    class ConfigObject(dict):
        """
        Represents configuration options' group, works like a dict
        """

        def __init__(self, *args, **kwargs):
            dict.__init__(self, *args, **kwargs)

        def __getattr__(self, name):
            return self[name]

        def __setattr__(self, name, val):
            self[name] = val

    if isinstance(config, dict):
        result = ConfigObject()
        for key in config:
            result[key] = config2object(config[key])
        return result
    else:
        return config


def load_defaults(defaults_file: list = []):
    """
    Load default configuration from a list of file.
    """
    cfg = Config("configs/default.yaml")
    # cfg = cfg.update_config(Config("configs/dataset.yaml"))
    for file in defaults_file:
        print(file)
        cfg = deep_update(cfg, Config(file))
    
    cfg = Opts(cfg).parse_args()
   
    cfg = load_enviroment_path(cfg)
    return cfg

#
# Created on Mon Dec 05 2022 by Nguyen Qui Vinh Quang
#
# @licensed: Computer Vision & Image Processing Lab
# @project: VehicleReid
#
import logging
import os
import sys


# 设置一个日志记录器，根据传递的参数控制是否输出到控制台是否保存到文件，以及设置日志的格式和级别。
def setup_logger(name, save_dir, distributed_rank):
    # 创建一个名为name的日志记录器对象
    logger = logging.getLogger(name)
    # 设置日志记录器的日志级别为DEBUG，将记录所有级别的日志信息。
    logger.setLevel(logging.DEBUG)
    # 如果大于0，表示当前进程不是主进程，那么将不记录日志，直接返回日志记录器对象。
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    # 创建一个用于将日志信息输出到控制台的日志处理器（handler），并将输出流指定为标准输出流sys.stdout。
    ch = logging.StreamHandler(stream=sys.stdout)
    # 设置控制台日志处理器的日志级别为DEBUG
    ch.setLevel(logging.DEBUG)
    # 创建一个日志格式化对象，用于指定日志信息的显示格式
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    # 将上述创建的格式化对象应用到控制台日志处理器上
    ch.setFormatter(formatter)
    # 将控制台日志处理器添加到日志记录器中，以便将日志信息输出到控制台
    logger.addHandler(ch)

    if save_dir:
        # 创建一个用于将日志信息写入文件的日志处理器，指定日志文件的路径为save_dir目录下的"log.txt"，并以追加模式（"a"）打开文件
        fh = logging.FileHandler(os.path.join(save_dir, "log.txt"), mode="a")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

# vim: expandtab:ts=4:sw=4
import numpy as np


class Detection(object):
    """
    单个图像中的边界框检测
    This class represents a bounding box detection in a single image.

    Parameters
    ----------
    tlwh : array_like
        Bounding box in format `(x, y, w, h)`.
    confidence : float
        Detector confidence score.
    feature : array_like
        A feature vector that describes the object contained in this image.

    Attributes
    ----------
    tlwh : ndarray 表示边界框的位置和尺寸的数组
        Bounding box in format `(top left x, top left y, width, height)`.
    confidence : ndarray 检测器的置信度分数
        Detector confidence score.
    feature : ndarray | NoneType 表示帧索引的整数
        A feature vector that describes the object contained in this image.

    """

    def __init__(self, tlwh, confidence, feature, frame_idx, color_hist=None):
        # 将输入的tlwh参数转换为NumPy数组，数据类型为浮点数。这里tlwh表示边界框的左上角坐标 (x, y) 以及宽度 w 和高度 h
        self.tlwh = np.asarray(tlwh, dtype=np.float)
        self.confidence = float(confidence)
        # 将输入的feature参数转换为NumPy数组，数据类型为32位浮点数
        self.feature = np.asarray(feature, dtype=np.float32)
        self.frame_idx = frame_idx
        # 颜色直方图，可选参数，默认为None
        self.color_hist = color_hist

    # 用于将边界框转换为(min x, min y, max x, max y)格式，即左上角和右下角坐标
    def to_tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        # 将ret数组中的后两个元素（宽度和高度）加上前两个元素（左上角的 x 和 y 坐标），从而得到右下角的坐标
        ret[2:] += ret[:2]
        return ret

    # 将边界框转换为(center x, center y, aspect ratio, height)格式，其中aspect ratio表示宽度与高度的比率
    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.tlwh.copy()
        # 将ret数组中的前两个元素（左上角的 x 和 y 坐标）加上宽度和高度的一半，从而得到边界框的中心坐标 (center x, center y)
        ret[:2] += ret[2:] / 2
        # 将ret数组中的第三个元素（宽度）除以第四个元素（高度），从而计算出宽高比（aspect ratio）
        ret[2] /= ret[3]
        return ret

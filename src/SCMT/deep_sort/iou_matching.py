# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import linear_assignment

# 定义了两函数，iou和iou_cost，用于计算边界框之间的交并比（Intersection over Union）作为跟踪匹配的成本

def iou(bbox, candidates):
    """计算边界框之间的交并比
    Computer intersection over union.

    Parameters
    ----------
    bbox : ndarray
        A bounding box in format `(top left x, top left y, width, height)`.
    candidates : ndarray 候选边界框的矩阵，每行表示一个边界框，格式与bbox相同
        A matrix of candidate bounding boxes (one per row) in the same format
        as `bbox`.

    Returns
    -------
    ndarray
        The intersection over union in [0, 1] between the `bbox` and each
        candidate. A higher score means a larger fraction of the `bbox` is
        occluded by the candidate.

    """
    # 从bbox中提取左上角坐标bbox_tl和右下角坐标 bbox_br
    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
    # 从candidates中提取所有候选边界框的左上角坐标
    candidates_tl = candidates[:, :2]
    # 从candidates中计算所有候选边界框的右下角坐标
    candidates_br = candidates[:, :2] + candidates[:, 2:]

    # tl和 br分别计算交集的左上角和右下角坐标
    # tl：左上角坐标，由bbox_tl和每个候选边界框的左上角坐标的最大值组成
    # br：右下角坐标，由bbox_br和每个候选边界框的右下角坐标的最小值组成
    tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
               np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
    br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
               np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
    # 计算交集的宽度和高度，通过右下角坐标减去左上角坐标，同时确保宽度和高度不小于零
    wh = np.maximum(0., br - tl)

    # 计算交集的面积，即宽度和高度的乘积
    area_intersection = wh.prod(axis=1)
    # 计算输入边界框 bbox 的面积
    area_bbox = bbox[2:].prod()
    # 计算候选边界框的面积，对每个候选边界框进行相同的操作
    area_candidates = candidates[:, 2:].prod(axis=1)
    # 计算交并比，返回值为一个数组，每个元素表示bbox与候选边界框之间的交并比
    return area_intersection / (area_bbox + area_candidates - area_intersection)


def iou_cost(tracks, detections, track_indices=None,
             detection_indices=None):
    """用于计算跟踪匹配的成本矩阵
    An intersection over union distance metric.

    Parameters
    ----------
    tracks : List[deep_sort.track.Track]
        A list of tracks.
    detections : List[deep_sort.detection.Detection]
        A list of detections.
    track_indices : Optional[List[int]]
        A list of indices to tracks that should be matched. Defaults to
        all `tracks`.
    detection_indices : Optional[List[int]]
        A list of indices to detections that should be matched. Defaults
        to all `detections`.

    Returns
    -------
    ndarray
        Returns a cost matrix of shape
        len(track_indices), len(detection_indices) where entry (i, j) is
        `1 - iou(tracks[track_indices[i]], detections[detection_indices[j]])`.

    """
    # 如果未提供track_indices，则默认为所有跟踪
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    # 如果未提供detection_indices，则默认为所有检测
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    # 初始化一个用于存储成本矩阵的全零矩阵，其维度由track_indices和detection_indices的长度确定
    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    # 使用嵌套循环来计算每对跟踪和检测之间的交并比成本，并将结果填充到cost_matrix中。
    # 如果跟踪距离上一次更新时间大于1（可能是跟踪丢失），则将成本设置为无穷大INFTY_COST
    for row, track_idx in enumerate(track_indices):
        if tracks[track_idx].time_since_update > 1:
            cost_matrix[row, :] = linear_assignment.INFTY_COST
            continue

        bbox = tracks[track_idx].to_tlwh()
        candidates = np.asarray([detections[i].tlwh for i in detection_indices])
        cost_matrix[row, :] = 1. - iou(bbox, candidates)
    # 返回cost_matrix，
    # 其中每个元素(i, j)表示1 - iou(tracks[track_indices[i]], detections[detection_indices[j]])，即1减去交并比，作为匹配的成本
    return cost_matrix

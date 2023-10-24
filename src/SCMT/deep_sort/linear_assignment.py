# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment
#from scipy.optimize import linear_sum_assignment as linear_assignment 
from . import kalman_filter
from opts import opt

INFTY_COST = 1e+5


def min_cost_matching(
        distance_metric, max_distance, tracks, detections, track_indices=None,
        detection_indices=None):
    """该函数用于解决线性分配问题，即匹配跟踪与检测之间的关联
    Solve linear assignment problem.

    Parameters
    ----------
    distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
        这是一个函数参数，是一个可调用对象（Callable），用于计算跟踪与检测之间的关联成本。
        这个函数期望接收四个参数：跟踪列表、检测列表、跟踪索引列表和检测索引列表，并返回一个二维数组（矩阵），
        其中元素(i, j)表示给定跟踪索引中第i个跟踪与给定检测索引中第j个检测之间的关联成本
        The distance metric is given a list of tracks and detections as well as
        a list of N track indices and M detection indices. The metric should
        return the NxM dimensional cost matrix, where element (i, j) is the
        association cost between the i-th track in the given track indices and
        the j-th detection in the given detection_indices.
    max_distance : float
        门限阈值。如果关联的成本大于此值，将忽略该关联
        Gating threshold. Associations with cost larger than this value are
        disregarded.
    tracks : List[track.Track]
        当前时间步的预测跟踪列表
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        当前时间步的检测列表
        A list of detections at the current time step.
    track_indices : List[int]
        是一个可选参数，用于映射cost_matrix中的行到tracks中的跟踪。如果未提供此参数，将默认使用所有跟踪
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above).
    detection_indices : List[int]
        可选参数，用于映射cost_matrix中的列到detections中的检测。如果未提供此参数，将默认使用所有检测
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above).

    Returns
    -------
    (List[(int, int)], List[int], List[int])
        Returns a tuple with the following three entries:
        * A list of matched track and detection indices.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.

    """
    # 如果track_indices为None，则将track_indices设置为一个从0到tracks列表长度的数组，以映射到所有跟踪
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    # 如果detection_indices为None，则将detection_indices设置为一个从0到detections列表长度的数组，以映射到所有检测
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    # 检查是否没有要匹配的跟踪或检测
    if len(detection_indices) == 0 or len(track_indices) == 0:
        # 如果没有要匹配的内容，直接返回空列表，以及原始的跟踪和检测索引列表
        return [], track_indices, detection_indices  # Nothing to match.

    # 使用给定的distance_metric函数计算关联成本，得到一个成本矩阵
    cost_matrix = distance_metric(
        tracks, detections, track_indices, detection_indices)
    # 将大于max_distance的成本值设置为max_distance + 1e-5，即忽略门限外的关联
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5

    cost_matrix_ = cost_matrix.copy()

    # 使用线性分配算法（在linear_assignment模块中实现）来计算最佳关联。indices包含了匹配的跟踪和检测的索引
    indices = linear_assignment(cost_matrix_)

    matches, unmatched_tracks, unmatched_detections = [], [], []
    # 对检测索引进行迭代
    for col, detection_idx in enumerate(detection_indices):
        # 检查该列是否不在匹配的索引中
        if col not in indices[:, 1]:
            # 将该检测添加到未匹配的检测列表中
            unmatched_detections.append(detection_idx)
    # 对跟踪索引进行迭代
    for row, track_idx in enumerate(track_indices):
        # 检查该行是否不在匹配的索引中
        if row not in indices[:, 0]:
            # 将该跟踪添加到未匹配的跟踪列表中
            unmatched_tracks.append(track_idx)
    # 对匹配的索引进行迭代
    for row, col in indices:
        # 获取匹配的跟踪索引
        track_idx = track_indices[row]
        # 获取匹配的检测索引
        detection_idx = detection_indices[col]
        # 如果关联成本大于门限值将该跟踪和检测添加到未匹配的检测列表和未匹配的跟踪列表中
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            # 将该匹配的跟踪和检测的索引添加到匹配列表中
            matches.append((track_idx, detection_idx))
    # 返回匹配的跟踪和检测的索引列表，以及未匹配的跟踪和检测的索引列表
    return matches, unmatched_tracks, unmatched_detections


def matching_cascade(
        distance_metric, max_distance, cascade_depth, tracks, detections,
        track_indices=None, detection_indices=None):
    """运行级联匹配（Cascade Matching）来关联跟踪和检测
    Run matching cascade.

    Parameters
    ----------
    distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
        The distance metric is given a list of tracks and detections as well as
        a list of N track indices and M detection indices. The metric should
        return the NxM dimensional cost matrix, where element (i, j) is the
        association cost between the i-th track in the given track indices and
        the j-th detection in the given detection indices.
    max_distance : float
        Gating threshold. Associations with cost larger than this value are
        disregarded.
    cascade_depth: int
        # 表示级联的深度，通常应设置为最大跟踪年龄（maximum track age）。级联深度决定了运行级联匹配的次数
        The cascade depth, should be se to the maximum track age.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : Optional[List[int]]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above). Defaults to all tracks.
    detection_indices : Optional[List[int]]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above). Defaults to all
        detections.

    Returns
    -------
    (List[(int, int)], List[int], List[int])
        Returns a tuple with the following three entries:
        * A list of matched track and detection indices.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.

    """
    # 如果track_indices为None，将track_indices设置为一个从0到tracks列表长度的整数列表，以映射到所有跟踪
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    # 初始化未匹配的检测列表，将其设置为detection_indices的一个副本
    unmatched_detections = detection_indices
    matches = []
    # 检查opt.woC是否为真。opt.woC是一个选项（option），用于控制是否应用级联匹配
    if opt.woC:
        # 初始化一个新的跟踪索引列表 该列表将用于当前级联
        track_indices_l = [
            # 对跟踪索引进行迭代
            k for k in track_indices
            # 检查跟踪的time_since_update属性是否等于当前级联深度（level）。只有在一定年龄内的跟踪将用于当前级联
            # if tracks[k].time_since_update == 1 + level
        ]
        # 运行min_cost_matching函数，以匹配当前级联深度内的跟踪和检测，返回匹配结果、未匹配的跟踪和未匹配的检测
        matches_l, _, unmatched_detections = \
            min_cost_matching(
                distance_metric, max_distance, tracks, detections,
                track_indices_l, unmatched_detections)
        # 将当前级联深度的匹配结果添加到总的匹配列表中
        matches += matches_l
    else:
        # 对级联深度进行迭代，决定运行级联匹配的次数
        for level in range(cascade_depth):
            # 检查未匹配的检测是否为空。如果没有未匹配的检测，退出循环，因为不再需要匹配
            if len(unmatched_detections) == 0:  # No detections left
                break

            # 初始化一个新的跟踪索引列表 该列表将用于当前级联
            track_indices_l = [
                k for k in track_indices
                if tracks[k].time_since_update == 1 + level
            ]
            # 检查当前级联深度是否没有可用于匹配的跟踪
            if len(track_indices_l) == 0:  # Nothing to match at this level
                # 如果没有可用于匹配的跟踪，继续到下一个级联深度
                continue

            matches_l, _, unmatched_detections = \
                min_cost_matching(
                    distance_metric, max_distance, tracks, detections,
                    track_indices_l, unmatched_detections)
            matches += matches_l
    # 计算未匹配的跟踪，通过将所有的跟踪索引与匹配中的跟踪索引进行比较，得到未匹配的跟踪索引
    unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))
    return matches, unmatched_tracks, unmatched_detections

def gate_cost_matrix(
        cost_matrix, tracks, detections, track_indices, detection_indices,
        gated_cost=INFTY_COST, only_position=False, gating_threshold=50.0, add_identity=True):
    """
    用于在匹配跟踪目标和检测目标时，
    根据Kalman滤波器状态分布来验证和修正关联的代价矩阵（cost matrix）中的不合理条目。
    它有助于剔除那些基于Kalman滤波器预测的状态分布认为不合适的关联，从而提高跟踪的准确性
    Invalidate infeasible entries in cost matrix based on the state
    distributions obtained by Kalman filtering.

    Parameters
    ----------
    cost_matrix : ndarray
        The NxM dimensional cost matrix, where N is the number of track indices
        and M is the number of detection indices, such that entry (i, j) is the
        association cost between `tracks[track_indices[i]]` and
        `detections[detection_indices[j]]`.
    # 跟踪目标的列表，每个跟踪目标都有Kalman滤波器状态
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    # 检测目标的列表，每个检测目标通常包含位置、外观等信息
    detections : List[detection.Detection]
        A list of detections at the current time step.
    # 映射代价矩阵中的行到跟踪目标的索引
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above).
    # 映射代价矩阵中的列到检测目标的索引
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above).
    # 代价矩阵中的不合理条目将被设置为这个值，默认为一个极大的值（INFTY_COST）
    gated_cost : Optional[float]
        Entries in the cost matrix corresponding to infeasible associations are
        set this value. Defaults to a very large value.
    # 如果为True，只考虑状态分布的x和y位置信息，而不考虑其他维度（默认为False）
    only_position : Optional[bool]
        If True, only the x, y position of the state distribution is considered
        during gating. Defaults to False.

    Returns
    -------
    ndarray
        Returns the modified cost matrix.

    """
    # 检查only_position参数是否为False，因为函数中的代码只处理了only_position为False的情
    assert not only_position
    # gating_threshold = kalman_filter.chi2inv95[4]
    gating_threshold = gating_threshold
    # 从detection_indices中获取每个检测目标的位置信息（通常是x、y坐标、宽度和高度），形成一个NumPy数组
    measurements = np.asarray([detections[i].to_xyah() for i in detection_indices])
    for row, track_idx in enumerate(track_indices):
        # 获取跟踪目标的Kalman滤波器状态分布
        track = tracks[track_idx]
        # 使用Kalman滤波器的gating_distance方法计算每个检测目标与该跟踪目标之间的距离
        gating_distance = track.kf.gating_distance(track.mean, track.covariance, measurements, only_position, add_identity)
        # 使用Kalman滤波器的gating_distance方法计算每个检测目标与该跟踪目标之间的距离
        cost_matrix[row, gating_distance > gating_threshold] = gated_cost
        # 如果启用了opt.MC（可能是一种参数设置），将代价矩阵的每一行进行加权平均，
        # 其中权重是由距离信息和opt.MC_lambda（另一个参数）控制的。这是一个额外的操作，用于调整代价矩阵
        if opt.MC:
            cost_matrix[row] = opt.MC_lambda * cost_matrix[row] + (1 - opt.MC_lambda) *  gating_distance

    return cost_matrix

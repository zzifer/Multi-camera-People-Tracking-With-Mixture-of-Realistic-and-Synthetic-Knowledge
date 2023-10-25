# vim: expandtab:ts=4:sw=4
import numpy as np
from deep_sort.kalman_filter import KalmanFilter
from opts import opt

class TrackState:
    """
    枚举类型，表示单目标跟踪的状态
    Tentative（待确认）：新创建的跟踪目标初始状态，需要收集足够的证据后才能确认
    Confirmed（已确认）：跟踪目标已确认存在
    Deleted（已删除）：跟踪目标不再存在，标记为删除，将被从活动跟踪集合中移除
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    表示一个单目标跟踪，包含了跟踪目标的状态信息和相关属性
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.

    """

    def __init__(self, bbox, track_id, n_init, max_age,
                 detection):
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.state = TrackState.Tentative
        self.features = []
        self.color_hists = [] 
        # if feature is not None:
        #     feature /= np.linalg.norm(feature)
        #     self.features.append(feature)

        self.scores = []
        # if score is not None:
        #     self.scores.append(score)

        self._n_init = n_init
        self._max_age = max_age

        self.kf = KalmanFilter()

        self.mean, self.covariance = self.kf.initiate(bbox)
        
        # add by ljc
        self.storage =  []
        self.delete = False
        if detection is not None:
            feature = detection.feature / np.linalg.norm(detection.feature)
            self.features.append(feature)
            self.scores.append(detection.confidence)
            self.color_hists.append(detection.color_hist)
            self.storage.append(detection)


    def to_tlwh(self):
        """获取当前位置信息，并以边界框格式返回 (top left x, top left y, width, height)
        Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """获取当前位置信息，并以边界框格式返回 (min x, min y, max x, max y)
        Get current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self):
        """使用卡尔曼滤波器进行状态预测
        Propagate the state distribution to the current time step using a
        Kalman filter prediction step.
        """
        self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    @staticmethod
    def get_matrix(dict_frame_matrix, frame):
        eye = np.eye(3)
        # 从 dict_frame_matrix 字典中获取与给定 frame 对应的矩阵
        matrix = dict_frame_matrix[frame]
        dist = np.linalg.norm(eye - matrix)
        if dist < 100:
            return matrix
        else:
            return eye

    def update(self, detection):
        """执行卡尔曼滤波器的测量更新步骤，并更新特征缓存
        Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        detection : Detection
            The associated detection.

        """
        self.mean, self.covariance = self.kf.update(self.mean, self.covariance, detection.to_xyah(), detection.confidence)
        # self.mean[:4] = detection.confidence * detection.to_xyah() + (1 - detection.confidence) * self.mean[:4]
        # self.mean[:4] = detection.to_xyah()

        feature = detection.feature / np.linalg.norm(detection.feature)
        if opt.EMA:
            smooth_feat = opt.EMA_alpha * self.features[-1] + (1 - opt.EMA_alpha) * feature
            smooth_feat /= np.linalg.norm(smooth_feat)
            self.features = [smooth_feat]
            smooth_color_hist = opt.EMA_alpha * self.color_hists[-1] + (1 - opt.EMA_alpha) * detection.color_hist
            smooth_color_hist /= np.linalg.norm(smooth_color_hist)
            self.color_hists = [smooth_color_hist]
        else:
            self.features.append(feature)
            self.color_hists.append(detection.color_hist)

        self.hits += 1
        self.time_since_update = 0
        # detection.tlwh = detection.confidence * detection.tlwh + (1 - detection.confidence) * self.to_tlwh()
        # if self.hits > 10 and detection.confidence < 0.3:
        #     detection.tlwh = self.to_tlwh()
        self.storage.append(detection)
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        """将该跟踪目标标记为未匹配（在当前时间步没有关联的检测）
        Mark this track as missed (no association at the current time step).
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted

    def is_still(self, iou_thresh=0.95):
        """返回True，如果该跟踪目标不再移动，基于与前一帧的交叠比例来判断
        Return True if the track is not moving"""
        if len(self.storage) < 2:
            return False
        det1 = self.storage[-1]
        det2 = self.storage[-2]
        ltx1 = det1.tlwh[0]
        lty1 = det1.tlwh[1]
        rdx1 = det1.tlwh[0] + det1.tlwh[2]
        rdy1 = det1.tlwh[1] + det1.tlwh[3]
        ltx2 = det2.tlwh[0]
        lty2 = det2.tlwh[1]
        rdx2 = det2.tlwh[0] + det2.tlwh[2]
        rdy2 = det2.tlwh[1] + det2.tlwh[3]

        W = min(rdx1, rdx2) - max(ltx1, ltx2)
        H = min(rdy1, rdy2) - max(lty1, lty2)
        cross = W * H

        if(W <= 0 or H <= 0):
            return 0

        SA = (rdx1 - ltx1) * (rdy1 - lty1)
        SB = (rdx2 - ltx2) * (rdy2 - lty2)
        if min(SA, SB) <= 0:
            return 0
        iou = cross / min(SA, SB)
        if iou > iou_thresh:
            return True
        else:
            return False


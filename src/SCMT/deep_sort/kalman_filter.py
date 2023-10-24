# vim: expandtab:ts=4:sw=4
"""
@Filename: kalman_filter.py
@Discription: Kalman Filter
"""

import numpy as np
import scipy.linalg
from opts import opt
"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
# 定义了一个字典chi2inv95，包含了卡方分布的0.95分位数，用于Mahalanobis门限
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


class KalmanFilter(object):
    """用于实现卡尔曼滤波器，用于跟踪图像空间中的边界框
    A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space

        x, y, a, h, vx, vy, va, vh

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).

    """

    def __init__(self):
        ndim, dt = 4, 1.

        # Create Kalman filter model matrices.
        # 运动矩阵，表示状态转移模型，描述了状态在时间步进时如何变化
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        # 更新矩阵，表示状态观测模型，用于将状态映射到测量空间
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        # self._std_weight_position 和 self._std_weight_velocity：控制模型不确定性的权重
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        """用于从未关联的测量中创建一个新的跟踪
        Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.

        """
        # 提取测量中的位置信息
        mean_pos = measurement
        # 初始化速度信息为零
        mean_vel = np.zeros_like(mean_pos)
        # 将位置和速度信息合并为状态向量
        mean = np.r_[mean_pos, mean_vel]

        # 初始化状态的不确定性权重
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]]
        # 构建状态的协方差矩阵
        covariance = np.diag(np.square(std))
        # 返回新的状态均值和协方差矩阵
        return mean, covariance

    def predict(self, mean, covariance):
        """定义了卡尔曼滤波器的预测方法，用于预测下一时刻的状态
        Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.

        """
        # 初始化位置和速度的不确定性权重
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]]
        # 构建状态的协方差矩阵
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        # 根据状态转移模型更新状态均值
        mean = np.dot(self._motion_mat, mean)
        # 根据状态转移模型和协方差矩阵更新协方差矩阵
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance, confidence=.0):
        """定义了卡尔曼滤波器的状态映射方法，用于将状态映射到测量空间
        Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        confidence: (dyh) 检测框置信度
        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.

        """
        # 初始化状态的不确定性权重
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]]

        if opt.NSA:
            std = [(1 - confidence) * x for x in std]

        # 构建状态的协方差矩阵
        innovation_cov = np.diag(np.square(std))

        # 根据观测模型更新状态均值
        mean = np.dot(self._update_mat, mean)
        # 根据观测模型和协方差矩阵更新协方差矩阵
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement, confidence=.0):
        """定义了卡尔曼滤波器的状态更新方法，用于校正状态
        Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.
        confidence: (dyh)检测框置信度
        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.

        """
        # 获取状态的映射结果
        projected_mean, projected_cov = self.project(mean, covariance, confidence)

        # 使用Cholesky分解计算Cholesky因子，用于解线性方程
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        # 计算卡尔曼增益
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        # 计算测量和映射均值之间的创新
        innovation = measurement - projected_mean

        # 根据卡尔曼增益更新状态均值
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        # 根据卡尔曼增益和协方差矩阵更新协方差矩阵
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False, add_identity=True):
        """定义了卡尔曼滤波器的门限距离计算方法，用于衡量状态和测量之间的距离
        Compute gating distance between state distribution and measurements.

        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.

        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.

        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.

        """
        # 根据输入参数获取映射后的状态均值和协方差矩阵
        mean, covariance = self.project(mean, covariance)

        # 如果为True，只计算位置信息的距离
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        # 如果为True，将单位矩阵添加到协方差矩阵
        if add_identity:
            # 单位矩阵
            identity = np.identity(4)
            alpha = 50.0
            covariance += alpha * identity

        # 使用Cholesky分解计算Cholesky因子。
        cholesky_factor = np.linalg.cholesky(covariance)
        # 计算测量和状态均值之间的差距
        d = measurements - mean
        # 计算马氏距离的平方
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False,
            overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)
        # 返回距离的数组，每个元素表示状态与测量之间的距离
        return squared_maha

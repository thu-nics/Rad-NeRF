import os
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import splprep, splev

def _process_points3d(points3d):
    pts3d_world = []
    for i, pt_idx in enumerate(points3d):
        pts3d_world.append(points3d[pt_idx].xyz.tolist())
    return np.array(pts3d_world)

def get_bbox_from_points(points: np.ndarray, ignore_percentile=0):
    """_summary_

    Args:
        points (_type_): [N, d]
        ignore_percentile: 忽略首尾比较稀疏的点

    Returns:
        _type_: _description_
    """
    d = points.shape[1]
    points = np.array(points).T

    bbox = np.zeros((2, d), np.float64)
    if points.size == 0:
        return bbox
    for i in range(d):
        bbox[:, i] = [np.percentile(points[i], ignore_percentile), np.percentile(points[i], 100 - ignore_percentile)]
    center = np.mean(bbox, axis=0)
    scene_range = bbox[1] - bbox[0]
    scene_range *= 1.05
    bbox[0] = center - scene_range / 2
    bbox[1] = center + scene_range / 2

    return bbox

def filter_outliers_by_boxplot(points, dims=tuple()):
    """https://en.wikipedia.org/wiki/Box_plot

    Args:
        points (_type_): [N, dim]
        dims: only filer outliers based on given dims
    """
    dim = points.shape[1]
    bbox = np.zeros((2, dim))
    if len(dims) == 0:
        dims = list(range(dim))

    for axis_idx in range(dim):
        if axis_idx in dims:
            Q3 = np.percentile(points[:, axis_idx], 75)
            Q1 = np.percentile(points[:, axis_idx], 25)
            IQR = Q3 - Q1
            min_value = Q1 - 1.5 * IQR
            max_value = Q3 + 1.5 * IQR
        else:
            min_value = np.min(points[:, axis_idx])
            max_value = np.max(points[:, axis_idx])
        bbox[:, axis_idx] = (min_value, max_value)

    mask = np.bitwise_and(points >= bbox[:1], points <= bbox[1:])
    mask = mask.all(axis=-1)
    points_inliers = points[mask]
    return points_inliers

def normalize_points(points, bbox):
    points = (points-bbox[0])/(bbox[1]-bbox[0])
    points = points.clip(0.0, 1.0)
    return points

class PoseRigRail:
    def __init__(self, key_pose_list, n_samples) -> None:
        self.key_pose_list = key_pose_list
        self.n_samples = n_samples

    def _interpolate_rotation(self, key_rots, n_samples):
        """
        :param rots: list of Rotation
        :param n_samples: number of samples
        :return:
        """
        N_key_pose = len(key_rots)
        key_times = list(range(N_key_pose))
        slerp = Slerp(key_times, key_rots)

        times = np.linspace(0, N_key_pose-1, n_samples)
        rots = slerp(times)
        return rots

    def _interpolate_translation(self, key_ts, n_samples):
      """
      :param ts: list of translation(3,)
      :param n_samples: number of samples
      :return:
      """
      N_key_ts = len(key_ts)
      key_ts = np.array(key_ts).transpose()
      tck, u = splprep(key_ts, s=10, k=min(N_key_ts-1, 3))
      # 控制点
    #   x_knots, y_knots, z_knots = splev(tck[0], tck)
      u_fine = np.linspace(0,1,n_samples)
      x_fine, y_fine, z_fine = splev(u_fine, tck)
      return np.array([x_fine, y_fine, z_fine]).transpose()


    def _interploate_se3(self, key_pose_list, n_samples):
        """
        :param pose_list: list of 4*4 mat
        :param n_samples: number of samples
        :return:
        """
        pose_len = len(key_pose_list)
        key_pose_list = np.array(key_pose_list)
        key_rots = []
        key_ts = []
        cnt = 0
        for pose3x4 in key_pose_list:
            rot = Rotation.from_matrix(pose3x4[:3, :3])

            cnt+=1
            key_ts.append(pose3x4[:3, 3])
            if cnt==1 or cnt==pose_len:
                key_rots.append(pose3x4[:3, :3])
        key_rots = np.array(key_rots)
        key_rots = Rotation.from_matrix(key_rots)

        rots = self._interpolate_rotation(key_rots, n_samples)
        ts = self._interpolate_translation(key_ts, n_samples)
        poses = []
        for i in range(n_samples):
            pose = np.eye(4)
            pose[:3, :3] = rots[i].as_matrix()
            pose[:3, 3] = ts[i]
            poses.append(pose)
        return poses
    
    def get_pose_list(self):
        self.pose_list = self._interploate_se3(self.key_pose_list, self.n_samples)
        return self.pose_list

def inter_two_poses(pose_a, pose_b, alpha):
    ret = np.zeros([3, 4], dtype=np.float64)
    rot_a = Rotation.from_matrix(pose_a[:3, :3])
    rot_b = Rotation.from_matrix(pose_b[:3, :3])
    key_rots = Rotation.from_matrix(np.stack([pose_a[:3, :3], pose_b[:3, :3]], 0))
    key_times = [0, 1]
    slerp = Slerp(key_times, key_rots)
    rot = slerp(1. - alpha)
    ret[:3, :3] = rot.as_matrix()
    ret[:3, 3] = (pose_a * alpha + pose_b * (1. - alpha))[:3, 3]
    return ret


def inter_poses(ori_poses, n_out_poses, interval=5):
    key_poses = ori_poses[::interval]
    n_key_poses = len(key_poses)
    out_poses = []
    for i in range(n_out_poses):
        w = np.linspace(0, n_key_poses - 1, n_key_poses)
        w = np.exp(-(np.abs(i / n_out_poses * n_key_poses - w))**2)
        w = w + 1e-6
        w /= np.sum(w)
        cur_pose = key_poses[0]
        cur_w = w[0]
        for j in range(0, n_key_poses - 1):
            cur_pose = inter_two_poses(cur_pose, key_poses[j + 1], cur_w / (cur_w + w[j + 1]))
            cur_w += w[j + 1]

        out_poses.append(cur_pose)

    return np.stack(out_poses)
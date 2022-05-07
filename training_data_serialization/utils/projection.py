import numpy as np
import cv2
from copy import deepcopy
from numpy.lib.stride_tricks import as_strided


def cart2hom(pts_3d):
    """
    Input: nx3 points in Cartesian
    Output: nx4 points in Homogeneous by pending 1
    """
    n = pts_3d.shape[0]
    pts_3d_hom = np.hstack((deepcopy(pts_3d), np.ones((n, 1))))
    return pts_3d_hom


def get_condition(pts_on_img, pts_xyz, H, W, clip_distance, datasets_name=None):
    if datasets_name == "new_college":
        condition = (pts_on_img[:, 0] < W) & \
                    (pts_on_img[:, 0] >= 0) & \
                    (pts_on_img[:, 1] < H) & \
                    (pts_on_img[:, 1] >= 0) & \
                    (pts_xyz[:, 0] < clip_distance) & \
                    (pts_xyz[:, 1] < clip_distance)
    elif datasets_name == "kitti":
        condition = (pts_on_img[:, 0] < W) & \
                    (pts_on_img[:, 0] >= 0) & \
                    (pts_on_img[:, 1] < H) & \
                    (pts_on_img[:, 1] >= 0) & \
                    (pts_xyz[:, 0] > clip_distance)
    elif datasets_name == "nuscene":
        condition = (pts_on_img[:, 0] < W) & \
                    (pts_on_img[:, 0] >= 0) & \
                    (pts_on_img[:, 1] < H) & \
                    (pts_on_img[:, 1] >= 0) & \
                    (pts_xyz[:, 1] > clip_distance)
    else:
        raise "Not supported datasets_name: {}".format(datasets_name)

    return condition


def get_depth_map(pts_xyz, H, W, T, R, datasets_name, P=None, clip_distance=0, get_z_before_norm=False, norm_methods='group', lidar_range=None):
    assert norm_methods in ['group', 'lidar_range', 'lidar_range_and_lidar_depth'], "Normalization methods not supported."
    pts_xyz_hom = cart2hom(pts_xyz)
    pts_3d_ref = np.dot(pts_xyz_hom, np.transpose(deepcopy(T)))[:, :3]
    pts_2d = np.dot(pts_3d_ref, np.transpose(deepcopy(R)))
    if P is not None:
        pts_2d = cart2hom(pts_2d)
        pts_2d = np.dot(pts_2d, np.transpose(P))
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    pts_on_img = pts_2d[:, 0:3]
    if norm_methods == 'lidar_range_and_lidar_depth':
        # https://github.com/tinghuiz/SfMLearner/blob/master/kitti_eval/depth_evaluation_utils.py#L194
        pts_2d[:, 2] = pts_xyz[:, 0]
    # Filter Points not within range
    condition = get_condition(pts_on_img, pts_xyz, H, W, clip_distance, datasets_name)

    pts_cam_fov = pts_on_img[condition]

    x, y, z = pts_cam_fov[:, 0], pts_cam_fov[:, 1], pts_cam_fov[:, 2]

    phi_ = x.astype(int)
    phi_[phi_ < 0] = 0
    phi_[phi_ >= W] = W - 1

    theta_ = y.astype(int)
    theta_[theta_ < 0] = 0
    theta_[theta_ >= H] = H - 1

    depth_map = np.zeros((H, W, 1)) * 255.

    if norm_methods == 'group':
        try:
            zcam_normed = (deepcopy(z) - np.min(z))/ (np.max(z) - np.min(z))
        except:
            zcam_normed = np.zeros_like(z)
    else:
        assert lidar_range is not None, "lidar_range is None"
        try:
            zcam_normed = deepcopy(z) / float(lidar_range)
        except:
            zcam_normed = np.zeros_like(z)
        zcam_normed[zcam_normed > 1] = 1.

    depth_map[theta_, phi_, 0] = zcam_normed * 255.

    if get_z_before_norm:
        return depth_map, z
    return depth_map, None


def display_projected_img(pts_xyz, cam_fp, T, R, datasets_name, P=None, clip_distance=0, dense=False):
    cam = cv2.imread(cam_fp)
    H, W = cam.shape[:2]
    if dense:
        depth_map = get_dense_depth_map(pts_xyz=pts_xyz, H=H, W=W, T=T, R=R, P=P, datasets_name=datasets_name, clip_distance=clip_distance)
        cam = cv2.resize(cam, (depth_map.shape[1], depth_map.shape[0]))
    else:
        depth_map = get_depth_map(pts_xyz=pts_xyz, H=H, W=W, T=T, R=R, P=P, datasets_name=datasets_name, clip_distance=clip_distance)
    depth_map_vis = np.concatenate([depth_map, depth_map, depth_map], -1)
    overlay = cv2.addWeighted(cam.astype(np.uint8),0.4, depth_map_vis.astype(np.uint8), 1, -1)
    return overlay


def pool2d(A, kernel_size, stride, padding, pool_mode='max'):
    '''
    2D Pooling

    Parameters:
        A: input 2D array
        kernel_size: int, the size of the window
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'
    '''
    # Padding
    A = np.pad(A, padding, mode='constant')

    # Window view of A
    output_shape = ((A.shape[0] - kernel_size) // stride + 1,
                    (A.shape[1] - kernel_size) // stride + 1)
    kernel_size = (kernel_size, kernel_size)
    A_w = as_strided(A, shape=output_shape + kernel_size,
                     strides=(stride * A.strides[0],
                              stride * A.strides[1]) + A.strides)
    A_w = A_w.reshape(-1, *kernel_size)

    # Return the result of pooling
    if pool_mode == 'max':
        return A_w.max(axis=(1, 2)).reshape(output_shape)
    elif pool_mode == 'avg':
        return A_w.mean(axis=(1, 2)).reshape(output_shape)


def get_dense_depth_map(pts_xyz, H, W, T, R, datasets_name, P=None, kernel_size=8, stride=4, layers=1, clip_distance=0, debug=False, skip_pooling=False, get_z_before_norm=False, norm_methods='group', lidar_range=None):
    depth_map, z = get_depth_map(pts_xyz, H, W, T, R, datasets_name=datasets_name, P=P, clip_distance=clip_distance, get_z_before_norm=get_z_before_norm, norm_methods=norm_methods, lidar_range=lidar_range)
    if skip_pooling:
        return depth_map
    depth_map_max_pooled = pool2d(np.squeeze(depth_map, -1), kernel_size=kernel_size, stride=stride, padding=0, pool_mode='max')
    if layers > 1:
        for _ in range(1, layers):
            depth_map_max_pooled = pool2d(depth_map_max_pooled, kernel_size=kernel_size, stride=stride, padding=0, pool_mode='max')
    if debug:
        print("{} ~> {}".format(depth_map.shape, depth_map_max_pooled.shape))
        cv2.imwrite("depth_map_max_pooled.png", cv2.applyColorMap(depth_map_max_pooled.astype(np.uint8), cv2.COLORMAP_JET))

    if get_z_before_norm:
        return np.expand_dims(depth_map_max_pooled, -1), z

    return np.expand_dims(depth_map_max_pooled, -1)

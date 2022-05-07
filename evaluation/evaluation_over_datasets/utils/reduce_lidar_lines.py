import numpy as np


def reduce_lidar_lines(xyz_intensity, reduce_lidar_line_to, original_lines=64):
    if reduce_lidar_line_to == original_lines:
        return xyz_intensity
    velo_down = []
    pt_num = xyz_intensity.shape[0]
    down_Rate = original_lines / reduce_lidar_line_to
    line_num = int(pt_num / original_lines)

    for i in range(original_lines):
        if i % down_Rate == 0:
            for j in range(int(-line_num/2), int(line_num/2)):
                velo_down.append(xyz_intensity[i*line_num+j])
    data_reduced = np.array(velo_down)
    return data_reduced

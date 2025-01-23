import open3d as o3d
import numpy as np
import cv2

def depth_to_point_cloud(depth_image, intrinsic_matrix):
    # 获取深度图像尺寸
    height, width = depth_image.shape

    # 提取相机内参矩阵
    fx, fy, cx, cy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1], intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
    points = []

    # 遍历每个像素点，计算对应的三维空间坐标
    for v in range(height):
        for u in range(width):
            z = depth_image[v, u] / 1000.0  # 将毫米转换为米
            if z == 0:
                continue
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            points.append([x, y, z])

    # 将点列表转换为Open3D点云格式
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(np.array(points))

    return point_cloud
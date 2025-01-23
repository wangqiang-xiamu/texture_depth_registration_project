import open3d as o3d

def apply_icp(source, target, threshold=0.02):
    # 使用ICP算法进行点云配准，threshold为距离阈值
    icp = o3d.pipelines.registration.registration_icp(source, target, threshold)
    return icp.transformation
import cv2
import numpy as np
import os


def stitch_textures(textures):
    """
    拼接纹理图像，并计算拼接结果的权重图像和仿射变换矩阵。

    参数:
    - textures: 纹理图像列表（多个图像）

    返回:
    - stitched_texture: 拼接后的纹理图像（拼接结果）
    - weight_map: 拼接过程中每个像素的权重图像（每个像素的贡献度）
    - affine_matrices: 拼接过程中使用的仿射变换矩阵列表（图像之间的几何变换）
    """
    stitcher = cv2.Stitcher_create()  # 使用cv2.Stitcher_create()代替cv2.createStitcher()
    status, stitched_texture = stitcher.stitch(textures)  # 进行图像拼接

    if status != cv2.Stitcher_OK:
        raise Exception(f"图像拼接失败，错误码: {status}")

    affine_matrices = []
    weight_map = np.ones_like(stitched_texture)  # 默认每个像素的权重均为1（均匀权重图像）

    for i in range(len(textures) - 1):
        affine_matrix = compute_affine_matrix(textures[i], textures[i + 1])
        affine_matrices.append(affine_matrix)

    return stitched_texture, weight_map, affine_matrices


def compute_affine_matrix(src_img, dst_img):
    """
    计算从源图像到目标图像的仿射变换矩阵

    参数:
    - src_img: 源图像（拼接顺序中的前一个图像）
    - dst_img: 目标图像（拼接顺序中的下一个图像）

    返回:
    - affine_matrix: 仿射变换矩阵
    """
    orb = cv2.ORB_create()  # 创建ORB特征检测器
    kp1, des1 = orb.detectAndCompute(src_img, None)  # 提取源图像的关键点和描述符
    kp2, des2 = orb.detectAndCompute(dst_img, None)  # 提取目标图像的关键点和描述符

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # 创建暴力匹配器，NORM_HAMMING适合ORB特征
    matches = bf.match(des1, des2)  # 匹配源图像和目标图像的特征点

    # 从匹配结果中提取出源图像和目标图像的匹配点坐标
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # 计算仿射变换矩阵
    affine_matrix, _ = cv2.estimateAffine2D(src_pts, dst_pts)  # 使用cv2的函数估计仿射变换矩阵
    return affine_matrix


if __name__ == "__main__":
    # 检查文件路径
    image_paths = [f"../assets/texture{i}.jpg" for i in range(1, 5)]

    for path in image_paths:
        if not os.path.exists(path):
            print(f"图像文件未找到: {path}")
        else:
            print(f"图像文件存在: {path}")

    # 读取4张纹理图像
    textures = [cv2.imread(path) for path in image_paths]

    # 调用拼接函数
    stitched_texture, weight_map, affine_matrices = stitch_textures(textures)

    # 打印结果
    print("拼接图像大小:", stitched_texture.shape)
    print("仿射变换矩阵:", affine_matrices)

    # 显示拼接结果
    cv2.imshow("Stitched Texture", stitched_texture)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 保存拼接结果到文件
    save_path = "../assets/stitched_texture.jpg"  # 你想要保存的位置和文件名
    cv2.imwrite(save_path, stitched_texture)
    print(f"拼接结果已保存到: {save_path}")

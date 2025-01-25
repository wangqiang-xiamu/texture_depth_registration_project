import cv2
import numpy as np


def stitch_images(image1, image2):
    """
    拼接两张纹理图像并返回拼接后的图像。

    参数:
        image1: 第一张纹理图像。
        image2: 第二张纹理图像。

    返回:
        stitched_image: 拼接后的纹理图像。如果拼接失败，返回 None。
    """
    stitcher = cv2.Stitcher_create()
    status, stitched_image = stitcher.stitch([image1, image2])

    if status != cv2.Stitcher_OK:
        print("图像拼接失败！")
        return None
    else:
        print("拼接成功！")
        return stitched_image


def calculate_affine_transform(image1, image2):
    """
    计算两张图像之间的仿射变换矩阵。

    参数:
        image1: 第一张纹理图像。
        image2: 第二张纹理图像。

    返回:
        affine_matrix: 两张图像之间的仿射变换矩阵。
    """
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(image1, None)
    kp2, des2 = orb.detectAndCompute(image2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    points1 = np.array([kp1[m.queryIdx].pt for m in matches])
    points2 = np.array([kp2[m.trainIdx].pt for m in matches])

    affine_matrix, inliers = cv2.estimateAffine2D(points1, points2)

    return affine_matrix


def process_two_images(image1, image2):
    """
    处理两张图像，拼接并计算仿射变换矩阵。

    参数:
        image1: 第一张图像。
        image2: 第二张图像。

    返回:
        stitched_image: 拼接后的图像。
        affine_matrix: 仿射变换矩阵。
    """
    # 拼接两张图像
    stitched_image = stitch_images(image1, image2)
    if stitched_image is None:
        print("拼接失败，停止处理！")
        return None, None

    # 计算仿射变换矩阵
    affine_matrix = calculate_affine_transform(image1, image2)
    return stitched_image, affine_matrix


# 示例使用
if __name__ == "__main__":
    # 打印图像路径，确保路径正确
    image1_path = '../assets/fusion4.jpg'
    image2_path = '../assets/fusion3.jpg'

    print("image1 path:", image1_path)

    # 加载图像
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    # 检查图像是否加载成功
    if image1 is None or image2 is None:
        print("某些图像加载失败，请检查图像路径或文件是否存在。")
        exit()

    # 处理两张图像，拼接并计算仿射变换矩阵
    stitched_image, affine_matrix = process_two_images(image1, image2)

    # 保存拼接后的图像
    if stitched_image is not None:
        cv2.imwrite('../assets/stitched_image2.jpg', stitched_image)
        print("拼接图像已保存！")

    # 打印仿射变换矩阵
    if affine_matrix is not None:
        print("图像对的仿射变换矩阵:\n", affine_matrix)

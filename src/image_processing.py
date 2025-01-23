import cv2
import numpy as np

def stitch_images(img1, img2):
    # 创建ORB特征检测器
    orb = cv2.ORB_create()

    # 计算关键点和描述符
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # 使用暴力匹配器匹配描述符
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # 排序匹配点
    matches = sorted(matches, key=lambda x: x.distance)

    # 提取匹配点坐标
    points1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    points2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # 估算仿射变换矩阵
    matrix, _ = cv2.estimateAffine2D(points1, points2)

    # 使用仿射变换矩阵将第二张图像变换并拼接到第一张图像
    result = cv2.warpAffine(img2, matrix, (img1.shape[1] + img2.shape[1], img1.shape[0]))
    result[0:img1.shape[0], 0:img1.shape[1]] = img1

    return result, matrix
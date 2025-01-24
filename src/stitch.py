import cv2
import numpy as np


def stitch_images(img1, img2):
    # 使用ORB特征检测和描述符计算
    orb = cv2.ORB_create()
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


def get_match_score(img1, img2):
    # 计算匹配的分数作为图像拼接的优先级（如匹配点的数量或者匹配质量）
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    return len(matches)  # 这里简单地使用匹配点数量作为评分标准


def dynamic_stitch(images):
    # 假设输入的images是一个未排序的图像列表
    n = len(images)
    stitched_images = [images[0]]  # 从第一张图像开始
    remaining_images = images[1:]  # 剩余的图像列表

    while remaining_images:
        best_match_score = -1
        best_img = None
        best_index = -1

        # 尝试将剩余的图像与已拼接的图像进行匹配
        for i, img in enumerate(remaining_images):
            match_score = get_match_score(stitched_images[-1], img)
            if match_score > best_match_score:
                best_match_score = match_score
                best_img = img
                best_index = i

        # 拼接最匹配的图像
        stitched_img, _ = stitch_images(stitched_images[-1], best_img)
        stitched_images.append(stitched_img)
        remaining_images.pop(best_index)

    return stitched_images[-1]  # 返回最终拼接的结果
import cv2
import numpy as np


def stitch_images_with_blending(img1, img2):
    """
    使用SIFT特征匹配计算透视变换矩阵，进行图像拼接并加入渐变融合。
    """
    # 使用SIFT提取图像特征
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # 使用暴力匹配器（BFMatcher）进行特征匹配
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)

    # 排序匹配点
    matches = sorted(matches, key=lambda x: x.distance)

    # 获取匹配点的坐标
    points1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    points2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # 确保有足够的匹配点
    if len(matches) < 4:
        print(f"匹配点太少 ({len(matches)}), 无法计算变换矩阵")
        return None, None

    # 计算透视变换矩阵
    matrix, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)

    # 如果没有得到有效的矩阵，返回None
    if matrix is None:
        print("无法计算变换矩阵！")
        return None, None

    # 获取两张图像的尺寸
    height1, width1, _ = img1.shape
    height2, width2, _ = img2.shape

    # 计算拼接后的图像大小，确保包含所有图像区域
    result_width = width1 + width2  # 拼接图像的宽度
    result_height = max(height1, height2)  # 高度取最大值

    # 进行透视变换，将图像2变换到拼接图像中
    result = cv2.warpPerspective(img2, matrix, (result_width, result_height))

    # 将图像1直接放置在拼接图像的左侧
    result[0:height1, 0:width1] = img1

    # 渐变融合
    blended_result = alpha_blending(img1, img2, matrix, result)

    return blended_result, matrix


def alpha_blending(img1, img2, matrix, result):
    """
    在拼接结果中加入渐变融合，平滑边界过渡。
    """
    height1, width1, _ = img1.shape
    height2, width2, _ = img2.shape

    # 创建权重图（alpha mask），用于控制两张图像的透明度
    blend_mask = np.zeros((height2, width2), dtype=np.float32)
    blend_mask[:, :] = 1.0  # 默认全不透明

    # 确定拼接区域的范围
    for i in range(height2):
        for j in range(width2):
            # 基于x或y坐标在接缝处生成渐变效果
            blend_mask[i, j] = min(i / height2, 1.0)

    # 将图像和权重图进行融合
    for i in range(height2):
        for j in range(width2):
            alpha = blend_mask[i, j]
            # 获取两张图像的像素值
            pixel1 = img1[i, j] if i < height1 and j < width1 else np.zeros(3)
            pixel2 = result[i, j] if i < height2 and j < width2 else np.zeros(3)
            # 进行渐变融合
            result[i, j] = pixel1 * (1 - alpha) + pixel2 * alpha

    return result


def validate_stitching(img1, img2, output_path="stitched_texture_with_blending.jpg"):
    """
    拼接两张纹理图像并可视化结果，验证拼接效果并保存结果。
    """
    stitched_image, matrix = stitch_images_with_blending(img1, img2)  # 使用无缝拼接方法

    if stitched_image is None:
        print("图像拼接失败！")  # 如果拼接失败，打印错误信息
        return

    # 显示拼接结果图像
    cv2.imshow("Stitched Texture", stitched_image)

    # 打印变换矩阵
    print("变换矩阵：\n", matrix)

    # 保存拼接后的图像到指定路径
    cv2.imwrite(output_path, stitched_image)  # 保存图像

    print(f"拼接后的图像已保存为 {output_path}")

    # 等待用户按键关闭图像窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#TEST
# 获取图像
img1 = cv2.imread('../assets/texture1.jpg')  # 替换为图像1的路径
img2 = cv2.imread('../assets/texture2.jpg')  # 替换为图像2的路径
# 确保图像成功读取
if img1 is None or img2 is None:
    print("图像加载失败！请检查路径。")
else:
    # 验证纹理图像拼接并保存结果
    validate_stitching(img1, img2, output_path="stitched_texture_with_blending_result.jpg")
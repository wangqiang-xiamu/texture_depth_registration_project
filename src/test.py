import cv2
import numpy as np

# 读取四张图片
img1 = cv2.imread('../assets/texture1.jpg')
img2 = cv2.imread('../assets/texture2.jpg')
img3 = cv2.imread('../assets/texture3.jpg')
img4 = cv2.imread('../assets/texture4.jpg')

# 假设重叠区域的宽度
overlap_width = 100  # 你可以根据实际情况调整这个值

# 确保图像大小一致
img_height = max(img1.shape[0], img2.shape[0], img3.shape[0], img4.shape[0])
img_width = max(img1.shape[1], img2.shape[1], img3.shape[1], img4.shape[1])

img1 = cv2.resize(img1, (img_width, img_height))
img2 = cv2.resize(img2, (img_width, img_height))
img3 = cv2.resize(img3, (img_width, img_height))
img4 = cv2.resize(img4, (img_width, img_height))

# 定义加权平均拼接函数
def blend_images(img1, img2, overlap_width, direction='horizontal'):
    """
    :param img1: 第一张图片
    :param img2: 第二张图片
    :param overlap_width: 重叠区域的宽度
    :param direction: 拼接方向，'horizontal' 表示左右拼接，'vertical' 表示上下拼接
    :return: 拼接后的图像
    """
    if direction == 'horizontal':
        # 水平拼接，重叠区域加权
        alpha = np.linspace(1, 0, overlap_width)  # 权重从1到0
        img1_overlap = img1[:, -overlap_width:]
        img2_overlap = img2[:, :overlap_width]

        # 使用加权平均法合并重叠区域
        blended_overlap = np.zeros_like(img1_overlap, dtype=np.float32)
        for i in range(overlap_width):
            blended_overlap[:, i] = alpha[i] * img1_overlap[:, i] + (1 - alpha[i]) * img2_overlap[:, i]

        # 可以对重叠区域应用高斯模糊来平滑过渡
        blended_overlap = cv2.GaussianBlur(blended_overlap.astype(np.uint8), (15, 15), 0)

        # 去掉不重叠部分，进行拼接
        img1_non_overlap = img1[:, :-overlap_width]
        img2_non_overlap = img2[:, overlap_width:]

        # 拼接并返回
        result = np.hstack((img1_non_overlap, blended_overlap, img2_non_overlap))
        return result.astype(np.uint8)

    elif direction == 'vertical':
        # 垂直拼接，重叠区域加权
        alpha = np.linspace(1, 0, overlap_width)  # 权重从1到0
        img1_overlap = img1[-overlap_width:, :]
        img2_overlap = img2[:overlap_width, :]

        # 使用加权平均法合并重叠区域
        blended_overlap = np.zeros_like(img1_overlap, dtype=np.float32)
        for i in range(overlap_width):
            blended_overlap[i, :] = alpha[i] * img1_overlap[i, :] + (1 - alpha[i]) * img2_overlap[i, :]

        # 可以对重叠区域应用高斯模糊来平滑过渡
        blended_overlap = cv2.GaussianBlur(blended_overlap.astype(np.uint8), (15, 15), 0)

        # 去掉不重叠部分，进行拼接
        img1_non_overlap = img1[:-overlap_width, :]
        img2_non_overlap = img2[overlap_width:, :]

        # 拼接并返回
        result = np.vstack((img1_non_overlap, blended_overlap, img2_non_overlap))
        return result.astype(np.uint8)

# 步骤1：左右拼接 img1 和 img2，确保平滑过渡
result1 = blend_images(img1, img2, overlap_width, direction='horizontal')

# 保存步骤1的结果
cv2.imwrite('save1.jpg', result1)

# 步骤2：左右拼接 img4 和 img3，确保平滑过渡
result2 = blend_images(img4, img3, overlap_width, direction='horizontal')

# 保存步骤2的结果
cv2.imwrite('save2.jpg', result2)

# 步骤3：上下拼接 result1 和 result2，确保平滑过渡
final_result = blend_images(result1, result2, overlap_width, direction='vertical')

# 显示最终拼接结果
cv2.imshow("Blended Image", final_result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存最终结果
cv2.imwrite('final_blended_image.jpg', final_result)

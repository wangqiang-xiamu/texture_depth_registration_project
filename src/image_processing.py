import cv2
import numpy as np
import os

def stitch_textures(textures):
    stitcher = cv2.Stitcher_create()  # 使用cv2.Stitcher_create()代替cv2.createStitcher()
    status, stitched_texture = stitcher.stitch(textures)  # 进行图像拼接

    # 添加调试信息，帮助查看拼接失败的原因
    print(f"拼接状态: {status}")
    if status != cv2.Stitcher_OK:
        print("拼接失败，请检查图像质量、重叠区域和特征匹配等问题。")
        raise Exception(f"图像拼接失败，错误码: {status}")

    affine_matrices = []
    weight_map = np.ones_like(stitched_texture)  # 默认每个像素的权重均为1（均匀权重图像）

    for i in range(len(textures) - 1):
        affine_matrix = compute_affine_matrix(textures[i], textures[i + 1])
        affine_matrices.append(affine_matrix)

    return stitched_texture, weight_map, affine_matrices


def compute_affine_matrix(src_img, dst_img):
    orb = cv2.ORB_create()  # 创建ORB特征检测器
    kp1, des1 = orb.detectAndCompute(src_img, None)  # 提取源图像的关键点和描述符
    kp2, des2 = orb.detectAndCompute(dst_img, None)  # 提取目标图像的关键点和描述符

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # 创建暴力匹配器，NORM_HAMMING适合ORB特征
    matches = bf.match(des1, des2)  # 匹配源图像和目标图像的特征点

    # 从匹配结果中提取出源图像和目标图像的匹配点坐标
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    affine_matrix, _ = cv2.estimateAffine2D(src_pts, dst_pts)  # 使用cv2的函数估计仿射变换矩阵
    return affine_matrix


def load_images(image_paths):
    textures = []
    for path in image_paths:
        if not os.path.exists(path):
            print(f"图像文件未找到: {path}")
            continue  # 跳过此图像文件
        img = cv2.imread(path)
        if img is None:
            print(f"图像读取失败: {path}")
        else:
            textures.append(img)
            print(f"成功读取图像: {path}")
    return textures


if __name__ == "__main__":
    image_paths = [f"../assets/fusion{i}.jpg" for i in range(3, 5)]

    # 读取图像
    textures = load_images(image_paths)

    if len(textures) < 2:
        print("未读取足够的图像进行拼接")
    else:
        try:
            stitched_texture, weight_map, affine_matrices = stitch_textures(textures)
            print("拼接图像大小:", stitched_texture.shape)
            print("仿射变换矩阵:", affine_matrices)

            if stitched_texture is not None:
                cv2.imshow("Stitched Texture", stitched_texture)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                save_path = "../assets/stitched_texture.jpg"
                cv2.imwrite(save_path, stitched_texture)
                print(f"拼接结果已保存到: {save_path}")
            else:
                print("拼接结果无效，无法显示或保存")
        except Exception as e:
            print(f"拼接过程中发生错误: {e}")

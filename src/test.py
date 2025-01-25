import cv2
import numpy as np


def show_matches(image1, image2):
    """
    可视化两张图像之间的特征点匹配，并打印匹配点数量。

    参数:
        image1: 第一张图像。
        image2: 第二张图像。
    """
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(image1, None)
    kp2, des2 = orb.detectAndCompute(image2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # 打印匹配的特征点数量
    print(f"匹配点数量: {len(matches)}")

    img_matches = cv2.drawMatches(image1, kp1, image2, kp2, matches[:20], None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('Matches', img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def stitch_images(image1, image2):
    """
    拼接两张图像并返回拼接后的图像。

    参数:
        image1: 第一张图像。
        image2: 第二张图像。

    返回:
        stitched_image: 拼接后的图像。
    """
    print(f"拼接图像 {image1.shape} 和 {image2.shape}...")
    stitcher = cv2.Stitcher_create()
    status, stitched_image = stitcher.stitch([image1, image2])

    if status != cv2.Stitcher_OK:
        print(f"图像拼接失败！状态码: {status}")  # 打印失败的状态码
        return None
    else:
        print("拼接成功！")
        return stitched_image


def load_image(image_path):
    """
    加载图像并检查是否成功。

    参数:
        image_path: 图像文件路径。

    返回:
        image: 加载的图像。
    """
    print(f"image1 path: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"图像加载失败！请检查路径 {image_path} 是否正确。")
    else:
        print(f"图像加载成功！尺寸: {image.shape}")
    return image


def process_multiple_images(images):
    """
    逐步拼接多个图像。

    参数:
        images: 包含多张图像的列表。

    返回:
        stitched_image: 拼接后的图像。
    """
    if len(images) < 2:
        print("需要至少两张图像进行拼接。")
        return None

    stitched_image = images[0]  # 从第一张图像开始
    for i in range(1, len(images)):
        stitched_image = stitch_images(stitched_image, images[i])
        if stitched_image is None:
            print("拼接失败，停止处理！")
            break
    return stitched_image


# 示例使用
if __name__ == "__main__":
    # 假设你有四张纹理图像路径
    image_paths = [
        '../assets/fusion1.jpg',
        '../assets/fusion2.jpg',
        '../assets/fusion3.jpg',
        '../assets/fusion4.jpg'
    ]

    images = [load_image(image_path) for image_path in image_paths]

    # 过滤掉加载失败的图像
    images = [image for image in images if image is not None]

    if len(images) >= 2:
        # 拼接多张图像
        stitched_image = process_multiple_images(images)
        if stitched_image is not None:
            cv2.imwrite('stitched_image.jpg', stitched_image)
            print("拼接图像已保存！")
        else:
            print("拼接失败！")
    else:
        print("图像不足，无法拼接！")

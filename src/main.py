import cv2
import numpy as np
import open3d as o3d
from point_cloud import depth_to_point_cloud
from registration import apply_icp
from stitch import dynamic_stitch

def main():
    # 读取纹理图像
    img1 = cv2.imread('assets/texture1.jpg')
    img2 = cv2.imread('assets/texture2.jpg')
    img3 = cv2.imread('assets/texture3.jpg')
    img4 = cv2.imread('assets/texture4.jpg')

    # 图像列表
    images = [img1, img2, img3, img4]

    # 自动调整拼接顺序
    final_stitched_img = dynamic_stitch(images)

    # 显示最终拼接图像
    cv2.imshow("Final Stitched Image", final_stitched_img)

    # 保存最终拼接图像
    output_image_path = 'final_stitched_image.jpg'
    cv2.imwrite(output_image_path, final_stitched_img)  # 保存拼接后的图像

    print(f"最终拼接图像已保存为 {output_image_path}")

    # 等待用户按键关闭图像窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 读取深度图像并转换为点云
    depth_image1 = cv2.imread('assets/depth1.png', cv2.IMREAD_UNCHANGED)
    depth_image2 = cv2.imread('assets/depth2.png', cv2.IMREAD_UNCHANGED)
    depth_image3 = cv2.imread('assets/depth3.png', cv2.IMREAD_UNCHANGED)
    depth_image4 = cv2.imread('assets/depth4.png', cv2.IMREAD_UNCHANGED)

    # 假设相机内参矩阵
    intrinsic_matrix = np.array([[500, 0, 320],
                                 [0, 500, 240],
                                 [0, 0, 1]])

    # 生成点云
    point_cloud1 = depth_to_point_cloud(depth_image1, intrinsic_matrix)
    point_cloud2 = depth_to_point_cloud(depth_image2, intrinsic_matrix)
    point_cloud3 = depth_to_point_cloud(depth_image3, intrinsic_matrix)
    point_cloud4 = depth_to_point_cloud(depth_image4, intrinsic_matrix)

    # 使用ICP算法配准点云
    transformation1 = apply_icp(point_cloud1, point_cloud2)
    point_cloud1.transform(transformation1)

    transformation2 = apply_icp(point_cloud1, point_cloud3)
    point_cloud1.transform(transformation2)

    transformation3 = apply_icp(point_cloud1, point_cloud4)
    point_cloud1.transform(transformation3)

    # 可视化配准后的点云
    o3d.visualization.draw_geometries([point_cloud1])

if __name__ == "__main__":
    main()
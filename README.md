1任务概述
目标
A．总体目标
对纹理图像和深度图像进行拼接，实现无缝融合。
B．算法步骤
	1.实现纹理图像的拼接，并获得先验信息，包括权重图像和仿射变换矩阵。
	2.将深度图像转化成点云，应用权重ICP算法对点云进行配准，其中点云的权重由纹理拼接所得到的权重图像给出，初始矩阵由仿射变换矩阵给出。
	3.通过权重ICP算法得到的变换矩阵实现点云的精确配准。
预期效果
1.	实现点云的精确配准，在视觉上与纹理图像拼接结果保证一致性。
2.	点云配准得到的变换矩阵能和纹理拼接所得到的仿射变换矩阵相互印证（旋转矩阵和平移矩阵存在关联，其结果能相互印证）。

##python环境
```bash
	pip install -r requirements.txt
```
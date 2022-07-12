# from PIL import Image
# import numpy as np
# from matplotlib import pyplot as plt
#
# # 创建3D对象
# fig = plt.figure(figsize=(20, 8), dpi=100)
# ax = fig.add_subplot(111, projection='3d')
#
# # 读取图片
# img = Image.open('1.png')
# pix = img.load()
# # 获取图片长宽
# width = img.size[0]
# height = img.size[1]
# # 创建x,y轴的长度
# x = np.arange(0, width)
# y = np.arange(0, height)
# x, y = np.meshgrid(x, y)
# z = np.zeros(width * height)
# # 建立列表，后期使用
# color = []
# # 遍历长宽，获得每个像素点的RGB值
# for i in range(width):
#     for j in range(height):
#         # 转化rgb为相对占比的元组
#         rgb = tuple(np.array(pix[i, j]) / 255)
#         color.append(rgb)
#
# # 在z=0的平面上作图
#
# # 设置颜色，像素
# ax.scatter(x, y, z, c=color, alpha=1)
# plt.show()


import matplotlib.pyplot as plt
from plyfile import PlyData
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
plydata = PlyData.read('D:\dataset\Linemod_preprocessed\models\obj_09.ply')

class_to_bbox_3D = np.array([[-52.2146, -38.7038, -42.8485],
        [ 52.2146, -38.7038, -42.8485],
        [ 52.2146,  38.7038, -42.8485],
        [-52.2146,  38.7038, -42.8485],
        [-52.2146, -38.7038,  42.8485],
        [ 52.2146, -38.7038,  42.8485],
        [ 52.2146,  38.7038,  42.8485],
        [-52.2146,  38.7038,  42.8485]])

xlist = plydata['vertex']['x']
ylist = plydata['vertex']['y']
zlist = plydata['vertex']['z']
x_box = class_to_bbox_3D[:,0]
y_box = class_to_bbox_3D[:,1]
z_box = class_to_bbox_3D[:,2]

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(xlist, ylist, zlist)
ax.scatter(x_box, y_box, z_box)
plt.show()
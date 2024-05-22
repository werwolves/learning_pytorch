import cv2
import numpy as np
import random
# 设置图像的大小
image_width, image_height = 1000, 1000

# 创建一个白色的空白图像
grid_image = np.ones((image_height, image_width, 3), dtype=np.uint8) * 255

# 定义网格线的间隔
grid_spacing = 20


colors_bgr = [
    [255, 0, 0],    # 红色
    [0, 255, 0],    # 绿色
    [0, 0, 255],    # 蓝色
    [255, 255, 0],  # 黄色
    [255, 0, 255],  # 品红色
    [0, 255, 255],  # 青色
    [192, 192, 192],# 灰色
    [128, 0, 0],    # 棕色
    [255, 165, 0]   # 橙色
]


# 画水平网格线
for i in range(0, image_height, grid_spacing):
    color = random.choice(colors_bgr)
    print(f'color:{color}')
    cv2.line(grid_image, (0, i), (image_width, i), color, 1)

# 画垂直网格线
for i in range(0, image_width, grid_spacing):
    color = random.choice(colors_bgr)
    cv2.line(grid_image, (i, 0), (i, image_height), color, 1)

cv2.imwrite('test_img_grid.png',grid_image)
# 显示网格图像
cv2.imshow('Grid Image', grid_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
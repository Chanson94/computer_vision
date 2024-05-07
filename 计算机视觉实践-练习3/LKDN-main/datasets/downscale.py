import cv2
import os

# CUDA_VISIBLE_DEIVICES=9 python datasets/downscale.py

# 输入文件夹路径和输出文件夹路径
input_folder = './datasets/Set5/original/'
output_folder = './datasets/Set5/LRbicx4/'

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历文件夹下所有 PNG 图片
for filename in os.listdir(input_folder):
    if filename.endswith('.png'):
        # 读取原始图片
        img = cv2.imread(os.path.join(input_folder, filename))
        # 使用 Bicubic 下采样四倍
        img_upscaled = cv2.resize(img, None, fx=1/4, fy=1/4, interpolation=cv2.INTER_CUBIC)
        # 保存下采样后的图片
        output_filename = os.path.join(output_folder, filename)
        cv2.imwrite(output_filename, img_upscaled)

        print(f'{filename} 处理完成')

print('所有图片处理完成')
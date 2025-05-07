from PIL import Image
import numpy as np

def crop_image_with_mask(image_path, mask_path, output_path):
    """
    根据掩码对RGB图片进行裁剪。
    
    :param image_path: 原始RGB图片的路径
    :param mask_path: 掩码图片的路径（黑白二值图像）
    :param output_path: 裁剪后的图片保存路径
    """
    # 打开原始图片和掩码图片
    image = Image.open(image_path).convert("RGB")  # 确保图片是RGB模式
    mask = Image.open(mask_path).convert("L")      # 确保掩码是灰度模式

    # 检查图片和掩码的尺寸是否一致
    if image.size != mask.size:
        raise ValueError("图片和掩码的尺寸不一致！")

    # 将图片和掩码转换为NumPy数组
    image_array = np.array(image)
    mask_array = np.array(mask)

    # 创建一个与原始图片相同大小的黑色图片
    cropped_image_array = np.ones_like(image_array)*255
    import pdb;pdb.set_trace()
    # 遍历每个像素，根据掩码决定是否保留
    for x in range(image_array.shape[1]):  # 遍历宽度
        for y in range(image_array.shape[0]):  # 遍历高度
            #import pdb;pdb.set_trace()
            if not (mask_array[y, x] == 16 or mask_array[y, x] == 3):  # 如果掩码的像素值为255（白色），则保留该像素
                cropped_image_array[y, x] = image_array[y, x]

    # 将裁剪后的图片数组转换回Pillow图像
    cropped_image = Image.fromarray(cropped_image_array)

    # 保存裁剪后的图片
    cropped_image.save(output_path)
    print(f"裁剪后的图片已保存到 {output_path}")

# 示例用法
image_path = "data/train/close_jar/variation0/episodes/episode0/wrist_rgb/0.png"  # 替换为你的RGB图片路径
mask_path = "data/train/close_jar/variation0/episodes/episode0/wrist_mask/0.png"   # 替换为你的掩码路径
output_path = "Gamba/rob_test/cropped_image.png"  # 替换为保存路径

crop_image_with_mask(image_path, mask_path, output_path)
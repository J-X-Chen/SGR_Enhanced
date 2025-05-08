from PIL import Image

def resize_image(input_path, output_path, target_size=(512, 512)):
    """
    将图片放大到指定大小。
    :param input_path: 输入图片的路径
    :param output_path: 输出图片的路径
    :param target_size: 目标图片的大小 (宽, 高)
    """
    # 打开图片
    with Image.open(input_path) as img:
        # 打印原始图片大小
        print(f"Original image size: {img.size}")
        
        # 放大图片
        resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
        
        # 保存放大后的图片
        resized_img.save(output_path)
        print(f"Resized image saved to {output_path}")

# 示例用法
input_image_path = "./images0/10.png"  # 替换为你的输入图片路径
output_image_path = "./images512/10.png"  # 替换为你想要保存的路径

resize_image(input_image_path, output_image_path)
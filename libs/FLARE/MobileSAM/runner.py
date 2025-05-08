import torch
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import cv2
import numpy as np
from mobile_sam import SamAutomaticMaskGenerator
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


model_type = "vit_t"
sam_checkpoint = "./weights/mobile_sam.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"

mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mobile_sam.to(device=device)
mobile_sam.eval()
'''
predictor = SamPredictor(mobile_sam)
predictor.set_image('./app/assets/picture1.jpg')
masks, _, _ = predictor.predict(<input_prompts>)
'''

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

mask_generator = SamAutomaticMaskGenerator(mobile_sam)
image_path = './data/w1.png'
image = Image.open(image_path).convert("RGB")
image_np = np.array(image)
image_torch = torch.from_numpy(image_np).permute(2, 0, 1).float().unsqueeze(0) / 255.0
masks = mask_generator.generate(image_np)

plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show() 

direction = 'right'
#查找桌子的mask
height, width = image.size
for i, mask in enumerate(masks):
    if mask['area'] > 5000 and mask['point_coords'][0][1] > 0.5 * width:
        x1=mask['bbox'][0]
        y1_list = np.where(mask['segmentation'][:,x1] != False)[0]
        y1 = y1_list[np.random.randint(len(y1_list))]

        y2=mask['bbox'][1]
        x2_list = np.where(mask['segmentation'][y2] != False)[0]
        x2 = x2_list[np.random.randint(len(x2_list))]

        x3=mask['bbox'][0]+mask['bbox'][2]
        y3_list = np.where(mask['segmentation'][:,x3] != False)[0]
        y3 = y3_list[0] if direction == 'left' else y3_list[-1]

        y4=mask['bbox'][1]+mask['bbox'][3]
        x4_list = np.where(mask['segmentation'][y4] != False)[0]
        x4 = x4_list[0] if direction == 'left' else x4_list[-1]
        break

#计算桌面四边形
table_points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.int32)

# 创建一个与原图大小相同的空白 mask
table_mask = torch.zeros((height, width), dtype=torch.bool)

# 使用 OpenCV 填充多边形，并转换为 PyTorch 张量
table_mask_cv = cv2.fillPoly(np.zeros((height, width), dtype=np.uint8), [table_points], 1)
table_mask |= torch.from_numpy(table_mask_cv).to(torch.bool)

# 读取所有物体的 mask
#object_masks = [cv2.imread(f'object_mask_{i}.png', cv2.IMREAD_GRAYSCALE) for i in range(1, 4)]  # 假设有3个物体

# 初始化最终的合并 mask
final_mask = torch.zeros_like(table_mask)

# 设置重叠误差阈值
overlap_threshold_value = 20  # 例如，允许物体至少有 50% 的面积在桌子内

for i, mask in enumerate(masks):

    # 计算物体 mask 与桌子 mask 的重叠区域 
    object_mask=torch.tensor(mask['segmentation'])
    # 计算物体 mask 与桌子 mask 的重叠区域
    intersection = torch.logical_and(object_mask, table_mask)
    intersection_area = torch.sum(intersection)
    # 计算重叠比例
    #overlap_ratio = intersection_area / mask['area'] if mask['area'] > 0 else 0
    
    # 如果重叠比例大于阈值，保留重叠部分
    #if overlap_ratio >= overlap_threshold:
    if intersection_area > overlap_threshold_value:
        final_mask = torch.logical_or(final_mask, object_mask)
    
    
    #if i==2:
    #    import pdb; pdb.set_trace()
    #    break
mask_path = f'./log/mask.png'
mask_result = [image_np[:,:,j] * final_mask.numpy() for j in range(3)]
#mask_result = torch.concat(mask_result, axis=-1)
mask_result = np.stack(mask_result, axis=-1)
Image.fromarray(mask_result).save(mask_path)

Image.fromarray(table_mask.numpy()).save(f'./log/table_mask.png')
Image.fromarray(final_mask.numpy()).save(f'./log/final_mask_mask.png')
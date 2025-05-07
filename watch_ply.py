from plyfile import PlyData
import numpy as np

# 读取 PLY 文件
plydata = PlyData.read('0.ply')

# 查看文件中的所有元素和属性
for element in plydata.elements:
    print(f"Element: {element.name}")
    for property in element.properties:
        print(f"  Property: {property.name}")

    # 获取当前元素的数据
    data = element.data
    print(f"Data shape of {element.name}: {data.shape}")

    # 如果是顶点数据，可以进一步查看每个属性的值
    if element.name == 'vertex':
        print("\nVertex data:")
        for prop_name in data.dtype.names:
            print(f"  {prop_name}: {data[prop_name]}")
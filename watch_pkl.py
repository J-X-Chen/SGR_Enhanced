import pickle
import os

def view_pkl_file(file_path):
    """
    查看.pkl文件中的内容
    :param file_path: .pkl文件的路径
    """
    try:
        # 打开并读取.pkl文件
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            print("文件内容如下：")
            print(data)
        import pdb;pdb.set_trace()
        # 如果需要修改数据并保存到新的.pkl文件
        if True:
            # 修改数据
            data._observations = [data._observations[0]]
            # 确保目标文件夹存在
            output_dir = os.path.dirname('data/test/open_microwave/variation0/episodes/episode2/low_dim_obs.pkl')
            os.makedirs(output_dir, exist_ok=True)
            # 保存修改后的数据到新的.pkl文件
            with open('data/test/open_microwave/variation0/episodes/episode2/low_dim_obs.pkl', 'wb') as file:
                pickle.dump(data, file)

    except FileNotFoundError:
        print(f"错误：文件 {file_path} 未找到。")
    except pickle.UnpicklingError:
        print(f"错误：无法解序列化文件 {file_path}。")
    except Exception as e:
        print(f"发生错误：{e}")

# 示例：查看一个.pkl文件
pkl_file_path = 'data/test/open_microwave (copy 1)/variation0/episodes/episode2/low_dim_obs.pkl'  # 替换为你的.pkl文件路径
view_pkl_file(pkl_file_path)


#low_dim_obs中的len(data)就是生成数据集图片的个数
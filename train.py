import os
import sys
#os.environ["DISPLAY"] = ":99"  # headless" sudo apt-get install xvfb; Xvfb :99 -screen 0 1024x768x16 &
os.environ["DISPLAY"] = ":1"  # use 'ps aux | grep X' to find the port number
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import hydra
import random
import logging
from omegaconf import DictConfig, OmegaConf, ListConfig

import run_seed_fn
from helpers.utils import create_obs_config

import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')
from torch.multiprocessing import set_start_method, get_start_method

try:
    if get_start_method() != 'spawn':
        set_start_method('spawn', force=True)
except RuntimeError:
    print("Could not set start method to spawn")
    pass
'''
fork：
特点：使用 fork 方法时，子进程会复制父进程的内存空间，包括所有变量和数据。这种方法在创建子进程时速度较快，但可能会导致一些问题，如内存泄漏和死锁，尤其是在使用 CUDA 时。
适用场景：适用于不需要在子进程中使用 CUDA 的情况。

spawn：
特点：使用 spawn 方法时，子进程会重新初始化父进程的代码和数据。这种方法在创建子进程时速度较慢，但更安全，特别是在使用 CUDA 时。
适用场景：适用于需要在子进程中使用 CUDA 的情况。这是 PyTorch 推荐的启动方法，特别是在多 GPU 训练中。

forkserver：
特点：使用 forkserver 方法时，会先启动一个服务器进程，然后从服务器进程中派生子进程。这种方法结合了 fork 和 spawn 的优点，既避免了 fork 的内存泄漏问题，又比 spawn 更快。
适用场景：适用于需要在多进程环境中高效创建子进程的情况，特别是在多 GPU 训练中。
'''
current_directory = os.getcwd()


@hydra.main(config_name='config', config_path='conf') #这个装饰器用于指定配置文件的路径和名称。读config_path/config_name(不包括扩展名)
def main(cfg: DictConfig) -> None:
    cfg_yaml = OmegaConf.to_yaml(cfg) #将 cfg 对象转换为 YAML 格式的字符串，方便打印和记录
    logging.info('Config:\n' + cfg_yaml)

    os.environ['MASTER_ADDR'] = cfg.ddp.master_addr
    master_port = (random.randint(0, 3000) % 3000) + 27000
    os.environ['MASTER_PORT'] = str(master_port)

    # convert relative paths to absolute paths for different cwds
    log_cwd = os.getcwd()
    os.chdir(current_directory)
    cfg.replay.path = os.path.abspath(cfg.replay.path)
    cfg.rlbench.demo_path = os.path.abspath(cfg.rlbench.demo_path)
    os.chdir(log_cwd) #'/home/kasm-user/saving/sgr/logs/close_jar/SGR_pointnext-xl_seg/sgrv2-demos_5-iter_20000/seed0'

    cfg.rlbench.cameras = cfg.rlbench.cameras \
        if isinstance(cfg.rlbench.cameras, ListConfig) else [cfg.rlbench.cameras]
    obs_config = create_obs_config(cfg.rlbench.cameras,
                                   cfg.rlbench.camera_resolution,
                                   cfg.method.name) #libs/RLBench/rlbench/observation_config.py 下CameraConfig object(参数大全)
    
    cfg.rlbench.tasks = cfg.rlbench.tasks if isinstance(
        cfg.rlbench.tasks, ListConfig) else [cfg.rlbench.tasks]
    multi_task = len(cfg.rlbench.tasks) > 1

    log_cwd = os.getcwd()
    logging.info('CWD:' + log_cwd)

    if cfg.framework.start_seed >= 0:
        # seed specified
        start_seed = cfg.framework.start_seed
    elif cfg.framework.start_seed == -1 and \
            len(list(filter(lambda x: 'seed' in x, os.listdir(log_cwd)))) > 0:
        # unspecified seed; use largest existing seed plus one
        largest_seed = max([
            int(n.replace('seed', ''))
            for n in list(filter(lambda x: 'seed' in x, os.listdir(log_cwd)))
        ])
        start_seed = largest_seed + 1
    else:
        # start with seed 0
        start_seed = 0

    seed_folder = log_cwd
    os.makedirs(seed_folder, exist_ok=True)

    with open(os.path.join(seed_folder, 'config.yaml'), 'w') as f:
        f.write(cfg_yaml)

    weights_folder = os.path.join(seed_folder, 'weights') #eg. logs/close_jar/SGR_pointnext-xl_seg/sgrv2-demos_5-iter_20000/seed0/weights
    if os.path.isdir(weights_folder) and len(os.listdir(weights_folder)) > 0:
        weights = os.listdir(weights_folder)
        latest_weight = sorted(map(int, weights))[-1]
        if latest_weight >= cfg.framework.training_iterations:
            logging.info(
                'Agent was already trained for %d iterations. Exiting.' %
                latest_weight)
            sys.exit(0)

    logging.info('Starting seed %d.' % start_seed)

    world_size = cfg.ddp.num_devices
    mp.spawn(run_seed_fn.run_seed,#每个子进程中运行的函数
             args=(
                 cfg,
                 obs_config,
                 cfg.rlbench.cameras,
                 multi_task,
                 start_seed,
                 world_size,
             ), #args元组包含了要传递给 run_seed 函数的参数
             nprocs=world_size,#指定要启动的进程数量
             join=True)#主进程会等待所有子进程完成后再继续执行


if __name__ == '__main__':
    main()

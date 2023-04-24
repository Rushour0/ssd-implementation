import os
import numpy as np
import torch
from ..utils import colorstr

def display_gpu_info():
    gpus = []  # type: list[dict[str, str]]
    for gpu in [torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())]:
        gpus.append(
            {'name': gpu.name,
            'total_memory': gpu.total_memory / (1024 * 1024),
            'compute_capability': f'{gpu.major}.{gpu.minor}',
            'multi_processor_count': gpu.multi_processor_count,
            'is_integrated': gpu.is_integrated,
            'is_multi_gpu_board': gpu.is_multi_gpu_board,
            }
        )

    os.system('cls' if os.name == 'nt' else 'clear')
    print(colorstr(f'CUDA is available : {torch.cuda.is_available() }', 'white'))
    print(colorstr(f'PyTorch version : {torch.__version__}', 'white'))
    print(colorstr(f'Numpy version : {np.__version__}', 'white'))

    for gpu in gpus:
        print()
        print(colorstr('GPU info', 'white'))
        print(
            colorstr('GPU name :', 'blue'),
            colorstr(gpu["name"], 'cyan')
        )
        print(
            colorstr('GPU total memory :', 'blue'),
            colorstr(f'{gpu["total_memory"]} MB', 'cyan')
        )
        print(
            colorstr('GPU compute capability :', 'blue'),
            colorstr(gpu["compute_capability"], 'cyan')
        )
        print(
            colorstr('GPU multi processor count :', 'blue'),
            colorstr(gpu["multi_processor_count"], 'cyan')
        )
        print(
            colorstr('GPU is integrated :', 'blue'),
            colorstr(gpu["is_integrated"], 'cyan')
        )
        print(
            colorstr('GPU is multi gpu board :', 'blue'),
            colorstr(gpu["is_multi_gpu_board"], 'cyan')
        )
        print()

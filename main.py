import os
import numpy as np
import torch
from SSD_VGG16D.utils import colorstr

gpus = [] # type: list[dict[str, str]]
for gpu in [torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())]:
    gpus.append({'name': gpu.name,
                 'total_memory': gpu.total_memory,
                 'compute_capability': f'{gpu.major}.{gpu.minor}',
                 'multi_processor_count': gpu.multi_processor_count,
                 'is_integrated': gpu.is_integrated,
                 'is_multi_gpu_board': gpu.is_multi_gpu_board,
                 
                 })

os.system('cls' if os.name == 'nt' else 'clear')
print(colorstr(f'CUDA is available : {torch.cuda.is_available() }', 'white'))
print(colorstr(f'PyTorch version : {torch.__version__}', 'white'))
print(colorstr(f'Numpy version : {np.__version__}', 'white'))

for gpu in gpus:
    print()
    print(colorstr('GPU info', 'white'))
    print(f'GPU name : {gpu["name"]}')
    print(f'GPU total memory : {gpu["total_memory"]}')
    print(f'GPU compute capability : {gpu["compute_capability"]}')
    print(f'GPU multi processor count : {gpu["multi_processor_count"]}')
    print(f'GPU is integrated : {gpu["is_integrated"]}')
    print(f'GPU is multi gpu board : {gpu["is_multi_gpu_board"]}')
    print()
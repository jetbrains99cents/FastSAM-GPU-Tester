import torch
import warnings
warnings.filterwarnings('ignore', category=UserWarning) # Suppress the low compute capability warning here
print(f'Local PyTorch version: {torch.__version__}')
cuda_available = torch.cuda.is_available()
print(f'Local CUDA available: {cuda_available}')
print(f'Local CUDA device count: {torch.cuda.device_count()}')
device_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"
print(f'Local Device name: {device_name}')

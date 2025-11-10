import torch

print(torch.cuda.get_device_name(0))
print(torch.cuda.memory_allocated()/1e6, "MB allocated")
print(torch.cuda.memory_reserved()/1e6, "MB reserved")

# -*- coding: utf-8 -*-

import torch

print("Cuda is available: " + str(torch.cuda.is_available()))
print("Cuda devices found: " + str(torch.cuda.device_count()))
print("Current driver is sufficient: " + str(torch._C._cuda_isDriverSufficient()))



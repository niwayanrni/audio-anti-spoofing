import torch
from aasist import AASIST_Style

model = AASIST_Style()

x = torch.randn(8, 128, 128)

y = model(x)

print("Output shape:", y.shape)
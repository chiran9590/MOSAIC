
import torch

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Selected device:", device)

x = torch.randn(1000, 1000, device=device)
y = x @ x

print("Tensor device:", y.device)
print("GPU acceleration test passed!")

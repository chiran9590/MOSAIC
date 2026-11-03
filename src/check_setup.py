
import sys
import numpy as np
import pandas as pd
import torch

print("=" * 45)
print("MOSAIC — Environment Check")
print("=" * 45)

print("Python version:", sys.version.split()[0])
print("NumPy version:", np.__version__)
print("Pandas version:", pd.__version__)
print("PyTorch version:", torch.__version__)
print("GPU available:", torch.cuda.is_available())

# Create a small example order book
order_book = pd.DataFrame({
    "bid_price": [100.00, 99.99, 99.98],
    "bid_volume": [500, 300, 200],
    "ask_price": [100.02, 100.03, 100.04],
    "ask_volume": [400, 250, 150]
})

order_book["mid_price"] = (
    order_book["bid_price"] + order_book["ask_price"]
) / 2

order_book["spread"] = (
    order_book["ask_price"] - order_book["bid_price"]
)

print("\nExample order book:")
print(order_book)

print("\nMOSAIC environment is ready!")

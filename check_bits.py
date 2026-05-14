import sys
import numpy as np
from src.hamming import hamming74_encode, hamming74_decode

with open("data/message.txt", "r") as f:
    data_binary = f.read().strip()
message_bits = np.array(list(map(int, data_binary)))

message_bits_h = hamming74_encode(message_bits)

# We need to get recovered_bits from the notebook. 
# It's not easily available unless we extract it from the notebook variables, which we can't do because it's a closed process.
# However, we can just look at message_bits_h.

print("Original first 20 bits: ", message_bits[:20])
print("Encoded first 28 bits: ", message_bits_h[:28])


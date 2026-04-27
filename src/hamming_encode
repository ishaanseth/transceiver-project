import numpy as np

def hamming74_encode(bits):
    bits = np.array(bits).astype(int)

    # Pad to multiple of 4
    if len(bits) % 4 != 0:
        bits = np.pad(bits, (0, 4 - len(bits) % 4))

    encoded = []

    for i in range(0, len(bits), 4):
        d1, d2, d3, d4 = bits[i:i+4]

        # Parity bits
        p1 = (d1 + d2 + d4) % 2
        p2 = (d1 + d3 + d4) % 2
        p3 = (d2 + d3 + d4) % 2

        # ترتیب: [p1 p2 d1 p3 d2 d3 d4]
        encoded.extend([p1, p2, d1, p3, d2, d3, d4])

    return np.array(encoded)

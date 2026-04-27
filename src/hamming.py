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

        # Order: [p1 p2 d1 p3 d2 d3 d4]
        encoded.extend([p1, p2, d1, p3, d2, d3, d4])

    return np.array(encoded)



def hamming74_decode(bits):
    bits = np.array(bits).astype(int)
    decoded = []

    for i in range(0, len(bits), 7):
        block = bits[i:i+7]
        if len(block) < 7:
            continue

        p1, p2, d1, p3, d2, d3, d4 = block

        # Syndrome calculation
        s1 = (p1 + d1 + d2 + d4) % 2
        s2 = (p2 + d1 + d3 + d4) % 2
        s3 = (p3 + d2 + d3 + d4) % 2

        error_pos = s1 + 2*s2 + 4*s3

        # Correct single-bit error
        if error_pos != 0:
            block[error_pos - 1] ^= 1

        # Extract original data bits
        decoded.extend([block[2], block[4], block[5], block[6]])

    return np.array(decoded)

import numpy as np

# --------------------------------------------------
# Parity-check matrix (simple LDPC example)
# H * c^T = 0 (mod 2)
# Codeword format: [d1 d2 d3 p1 p2 p3]
# --------------------------------------------------

H = np.array([
    [1, 1, 0, 1, 0, 0],
    [0, 1, 1, 0, 1, 0],
    [1, 0, 1, 0, 0, 1]
])


# --------------------------------------------------
# LDPC ENCODER
# --------------------------------------------------
def ldpc_encode(bits):
    bits = np.array(bits).astype(int)

    # pad to multiple of 3
    if len(bits) % 3 != 0:
        bits = np.pad(bits, (0, 3 - len(bits) % 3))

    encoded = []

    for i in range(0, len(bits), 3):
        d1, d2, d3 = bits[i:i+3]

        # parity bits
        p1 = (d1 + d2) % 2
        p2 = (d2 + d3) % 2
        p3 = (d1 + d3) % 2

        # codeword
        encoded.extend([d1, d2, d3, p1, p2, p3])

    return np.array(encoded)


# --------------------------------------------------
# LDPC DECODER (Bit-Flipping Algorithm)
# --------------------------------------------------
def ldpc_decode(bits, max_iterations=10):
    bits = np.array(bits).astype(int)
    decoded = []

    for i in range(0, len(bits), 6):
        block = bits[i:i+6].copy()

        if len(block) < 6:
            continue

        for _ in range(max_iterations):

            # syndrome check
            syndrome = np.mod(H @ block.T, 2)

            # if valid codeword
            if np.all(syndrome == 0):
                break

            # count which bits are involved in failed checks
            error_count = np.zeros(6, dtype=int)

            for row in range(H.shape[0]):
                if syndrome[row] == 1:
                    for col in range(H.shape[1]):
                        if H[row, col] == 1:
                            error_count[col] += 1

            # flip most likely wrong bit
            flip = np.argmax(error_count)
            block[flip] ^= 1

        # extract data bits
        decoded.extend(block[:3])

    return np.array(decoded)



""" import numpy as np
import tensorflow as tf
from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder


# --------------------------------------------------
# LDPC ENCODER
# --------------------------------------------------
def ldpc_encode(bits, k=100, n=200):
    bits = np.array(bits).astype(np.float32)

    # Pad to multiple of k
    if len(bits) % k != 0:
        bits = np.pad(bits, (0, k - len(bits) % k))

    encoder = LDPC5GEncoder(k=k, n=n)

    encoded = []

    for i in range(0, len(bits), k):
        block = bits[i:i+k]

        block = tf.convert_to_tensor(block.reshape(1, k))

        codeword = encoder(block)

        encoded.extend(codeword.numpy().flatten())

    return np.array(encoded).astype(int)


# --------------------------------------------------
# LDPC DECODER
# --------------------------------------------------
def ldpc_decode(received_signal, k=100, n=200):
    received_signal = np.array(received_signal).astype(np.float32)

    encoder = LDPC5GEncoder(k=k, n=n)
    decoder = LDPC5GDecoder(encoder, hard_out=True)

    decoded = []

    for i in range(0, len(received_signal), n):
        block = received_signal[i:i+n]

        if len(block) < n:
            continue

        block = tf.convert_to_tensor(block.reshape(1, n))

        data = decoder(block)

        decoded.extend(data.numpy().flatten())

    return np.array(decoded).astype(int)

def ldpc_decode_hard(bits, max_iterations=10):
    bits = np.array(bits).astype(int)
    decoded = []

    for i in range(0, len(bits), 6):
        block = bits[i:i+6].copy()

        if len(block) < 6:
            continue

        for _ in range(max_iterations):

            # Syndrome check
            syndrome = np.mod(H @ block.T, 2)

            # If no error, stop
            if np.all(syndrome == 0):
                break

            # Count which bits participate in failed checks
            error_count = np.zeros(6, dtype=int)

            for row in range(H.shape[0]):
                if syndrome[row] == 1:
                    for col in range(H.shape[1]):
                        if H[row, col] == 1:
                            error_count[col] += 1

            # Flip most likely wrong bit
            flip_index = np.argmax(error_count)
            block[flip_index] ^= 1

        # Extract data bits
        decoded.extend(block[:3])

    return np.array(decoded)

 """


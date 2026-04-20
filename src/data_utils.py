import numpy as np

def load_bitstream(filepath):
    """Reads a text file of '0's and '1's into a numpy array of integers."""
    with open(filepath, 'r') as f:
        bit_string = f.read().strip()
    return np.array([int(b) for b in bit_string])

def apply_error_correction(bit_array, method="none"):
    """
    Placeholder for Step 2. 
    Currently passes the data through untouched.
    Later, your team can add Hamming(7,4) or Convolutional coding here.
    """
    if method == "none":
        return bit_array
    else:
        raise NotImplementedError("Error correction not yet implemented.")
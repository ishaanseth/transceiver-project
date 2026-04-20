import numpy as np

def generate_zadoff_chu(u, root_seq_len):
    """
    Generates a Zadoff-Chu sequence for time-domain synchronization.
    
    Parameters:
    u (int): Root index. Must be relatively prime to root_seq_len.
    root_seq_len (int): Length of the sequence (N_zc). Prime numbers (e.g., 63, 127) are best.
    
    Returns:
    np.ndarray: Complex Zadoff-Chu sequence.
    """
    n = np.arange(root_seq_len)
    
    # Calculate the sequence based on the standard CAZAC formula
    zc_seq = np.exp(-1j * np.pi * u * n * (n + 1) / root_seq_len)
    
    return zc_seq

def append_sync_sequence(data_symbols, sync_sequence):
    """
    Appends the time-domain sync sequence to the start of the transmission.
    """
    return np.concatenate((sync_sequence, data_symbols))
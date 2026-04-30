import numpy as np
FS = 44100            # Sample rate (Hz)
F_CARRIER = 10000  
SYMBOL_RATE = 6300  # Symbols per second (Baud)

# Calculate how many audio samples represent one symbol
# 44100 Hz / 8820 Baud = 5 samples per symbol
# 44100 Hz / 7350 Baud = 6 samples per symbol
# 44100 Hz / 6300 Baud = 7 samples per symbol
SAMPLES_PER_SYMBOL = FS // SYMBOL_RATE

M=4 #M value for M qam, Mpam etc
total_symbols=1000

#zadoff chu
len_zadoff_chu = 353
u_zadoff_chu = 7

#METHODS
pulse_method="SINC" # "SQUARE" or "SINC"
modulation_method="QAM" # "QAM" or "PAM"

#do i want to read or write to the message.txt file, true to read 
flag=True

#gap between zadoff and pilots
SYNC_GAP_SECONDS=0

#SPAN for the sinc symbols 
SPAN=4

NUM_PILOTS = 32

bit_string = "1001001001101000110101010010011111011000001110111010010111111001"

pilot_bits = np.array([int(b) for b in bit_string[:NUM_PILOTS]])
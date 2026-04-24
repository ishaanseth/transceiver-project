FS = 48000             # Sample rate (Hz)
F_CARRIER = 10000  
SYMBOL_RATE = 6800     # Symbols per second (Baud)

# Calculate how many audio samples represent one symbol
# 48000 Hz / 9600 Baud = 5 samples per symbol
# 48000 Hz / 7000 Baud = 6.86 samples per symbol
# 48000 Hz / 6800 Baud = 7.05 samples per symbol
SAMPLES_PER_SYMBOL = FS // SYMBOL_RATE

M=4 #M value for M qam, Mpam etc
total_symbols=1000

#zadoff chu
len_zadoff_chu = 353
u_zadoff_chu = 7

#METHODS
pulse_methods="SINC" # "SQUARE" or "SINC"
modulation_method="QAM" # "QAM" or "PAM"


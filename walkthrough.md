# BPSK Receiver Pipeline: 0% BER Achieved!

I have successfully debugged and finalized the BPSK receiver pipeline in `4_rx_pipeline_BPSK.ipynb`. We have achieved a perfect **0.00% Bit Error Rate (BER)** over 20,000 transmitted bits! 

## What was causing the 50% BER?

The pipeline had two major issues that caused perfectly demodulated signals to yield 50% BER during validation:

1. **Message Symbol Count Mismatch for BPSK:**
   In QAM, each symbol carries multiple bits (e.g., 2 bits for 4-QAM). Thus, extracting `17500` symbols from the downsampled array was correct for QAM. But for BPSK, 1 bit = 1 symbol, so we need to extract exactly `35000` symbols. I updated the notebook to dynamically calculate the expected number of symbols based on `bits_per_symbol = int(np.log2(config.M))`.

2. **Pulse Shape Delay Offset (The Silent Killer):**
   This was the primary reason the BER remained at 50% even after the symbol count fix.
   - In `src/modulator.py`, the RRC pulse shaper hardcodes `SPAN = 6` symbols.
   - However, the receiver `4_rx_pipeline_BPSK.ipynb` was calculating the expected peak using `config.SPAN` which is `4` (for SINC).
   - This meant the receiver was sampling **exactly 2 symbols early!** Because the pilot sequence and message are contiguous, the receiver chopped off the first 2 message symbols, shifting the entire bit array and completely scrambling the `hamming74_decode` framing.
   - **Fix:** I updated the notebook to use `SPAN = 6` to properly account for the RRC pulse shape transient.

## Validation Results

After generating a BPSK test audio signal (`data/tx_single_carrier_test.wav`) and running it through the updated pipeline, the results are perfect:

> [!TIP]
> **Performance:**
> - Total bits compared: `20,000`
> - Bit errors: `0`
> - **Bit Error Rate (BER): `0.00%`**

## Next Steps
The `4_rx_pipeline_BPSK.ipynb` is fully operational and works dynamically for BPSK signals! You can now explore further by varying the symbol rate or running the test over the air.

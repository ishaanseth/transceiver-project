import json
with open('notebooks/4_rx_pipeline_BPSK.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        for i, line in enumerate(source):
            if 'start = pilot_start_idx + 6 * L + best_offset' in line:
                replacement = '''
# Dynamically determine the correct SPAN to avoid offset bugs
if config.pulse_method == "RRC":
    expected_span = 6
elif config.pulse_method == "SINC":
    expected_span = 4
else: # SQUARE
    expected_span = 0

start = pilot_start_idx + expected_span * L + best_offset
'''
                source[i] = replacement
                break

with open('notebooks/4_rx_pipeline_BPSK.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

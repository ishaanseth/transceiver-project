#!/bin/bash
echo "Setting up Transceiver Project Environment..."
python3 -m venv venv
source venv/bin/activate
echo "Installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt
echo "Registering Jupyter Kernel..."
python -m ipykernel install --user --name=transceiver_env
echo "Setup Complete! Run 'source venv/bin/activate' then 'jupyter lab' to start."
@echo off
echo Setting up Transceiver Project Environment...
python -m venv venv
call venv\Scripts\activate.bat
echo Installing requirements...
pip install --upgrade pip
pip install -r requirements.txt
echo Registering Jupyter Kernel...
python -m ipykernel install --user --name=transceiver_env
echo Setup Complete! Run 'venv\Scripts\activate' then 'jupyter lab' to start.
pause
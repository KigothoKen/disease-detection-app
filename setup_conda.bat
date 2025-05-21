@echo off
echo Setting up Miniconda environment for TensorFlow...

:: Initialize conda for cmd.exe
call %USERPROFILE%\Miniconda3\Scripts\activate.bat

:: Create a new environment
echo Creating new conda environment 'pig_env'...
call conda create -y -n pig_env python=3.9

:: Activate the environment
call conda activate pig_env

:: Install TensorFlow and other required packages
echo Installing TensorFlow and required packages...
call conda install -y -c conda-forge tensorflow=2.10.0
call conda install -y -c conda-forge pillow numpy flask

:: Copy the model if needed
if not exist "models\detect.tflite" (
  echo Please ensure your model is at models\detect.tflite
)

echo.
echo Setup complete! 
echo.
echo To activate the environment and run the app:
echo 1. Open a new command prompt
echo 2. Run: %USERPROFILE%\Miniconda3\Scripts\activate.bat
echo 3. Run: conda activate pig_env
echo 4. Run: python app_tf_fallback.py
echo.
echo Press any key to exit...
pause > nul 
@echo off

cd project

echo [1/4] Compiling unpacker.cpp...
g++ unpacker.cpp -o unpacker.exe
if errorlevel 1 (
    echo Compilation failed.
    pause
    exit /b 1
)

echo [2/4] Running unpacker...
unpacker.exe
if errorlevel 1 (
    echo Unpacker failed.
    pause
    exit /b 1
)

echo [3/4] Generating NPY...
python npygenerator.py
if errorlevel 1 (
    echo NPY generation failed.
    pause
    exit /b 1
)

echo [4/4] Visualizing NPY...
python visualize_npy.py
if errorlevel 1 (
    echo Visualization failed.
    pause
    exit /b 1
)

echo.
echo Cleaning temporary files...

cd result
del /q "decoded.txt" 2>nul
del /q "events.npy" 2>nul

echo.
echo All tasks completed successfully.
pause
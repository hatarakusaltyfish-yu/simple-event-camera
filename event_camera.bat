@echo off

cd project

echo [1/3] Running event camera simulator...
python "main.py"
if errorlevel 1 (
    echo Event camera simulation failed.
    pause
    exit /b 1
)

echo [2/3] Compiling compression.cpp...
g++ compression.cpp -o compression.exe
if errorlevel 1 (
    echo Compilation failed.
    pause
    exit /b 1
)

echo [3/3] Compressing events...
compression.exe
if errorlevel 1 (
    echo Compression failed.
    pause
    exit /b 1
)

echo.
echo Cleaning temporary files...

cd result
del /q "all_events.txt" 2>nul

echo.
echo All tasks completed successfully.
pause
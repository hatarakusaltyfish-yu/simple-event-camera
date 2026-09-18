@echo off
cd project
g++ unpacker.cpp -o unpacker.exe
unpacker.exe
python npygenerator.py
python visualize_npy.py
cd result
del "decoded.txt"
del "events.npy"
pause
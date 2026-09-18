@echo off
cd project
python "main.py"
g++ compression.cpp -o compression.exe
compression.exe
cd result
del "all_events.txt"
pause
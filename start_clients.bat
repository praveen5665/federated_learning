@echo off
REM Windows Batch script to start multiple clients in separate windows
REM This script opens multiple command prompt windows, one for each client

echo Starting Federated Learning Clients...
echo.

REM Start Client 1
start "Client 1" cmd /k "python client.py 1"

REM Wait a moment
timeout /t 2 /nobreak >nul

REM Start Client 2
start "Client 2" cmd /k "python client.py 2"

REM Wait a moment
timeout /t 2 /nobreak >nul

REM Start Client 3
start "Client 3" cmd /k "python client.py 3"

REM Wait a moment
timeout /t 2 /nobreak >nul

REM Start Client 4
start "Client 4" cmd /k "python client.py 4"

REM Wait a moment
timeout /t 2 /nobreak >nul

REM Start Client 5
start "Client 5" cmd /k "python client.py 5"

echo.
echo All clients started in separate windows!
echo Make sure the server is running first (python server.py)
echo.
pause


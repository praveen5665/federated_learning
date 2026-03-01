@echo off
echo Starting Federated Learning Clients...
echo.

for /L %%i in (1,1,10) do (
    start "Client %%i" cmd /k "python client_autoencoder.py %%i"
    timeout /t 2 /nobreak >nul
)

echo.
echo All clients started in separate windows!
echo Make sure the server is running first (python server_autoencoder.py 10)
echo.
pause


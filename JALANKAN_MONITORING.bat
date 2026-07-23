@echo off
echo [1/3] Menjalankan API Model dan Prometheus Exporter...
start cmd /k "title API Exporter && cd "Monitoring dan Logging" && python 3.prometheus_exporter.py"

echo Tunggu 3 detik agar API siap...
timeout /t 3 /nobreak >nul

echo [2/3] Menjalankan Server Prometheus...
start cmd /k "title Prometheus Server && cd "Monitoring dan Logging\prometheus\prometheus-2.45.3.windows-amd64" && prometheus.exe --config.file=..\..\2.prometheus.yml"

echo Tunggu 5 detik agar Prometheus siap membaca metrik...
timeout /t 5 /nobreak >nul

echo [3/3] Mengirimkan data traffic prediksi secara otomatis...
start cmd /k "title Traffic Generator && cd "Monitoring dan Logging" && python send_requests.py"

echo.
echo ========================================================
echo SEMUA SISTEM BERJALAN!
echo ========================================================
echo 1. Sekarang buka Grafana di browser (http://localhost:3000)
echo 2. Buka Prometheus di browser (http://localhost:9090)
echo 3. Tunggu grafik/angka bergerak naik, lalu ambil screenshot!
echo.
pause

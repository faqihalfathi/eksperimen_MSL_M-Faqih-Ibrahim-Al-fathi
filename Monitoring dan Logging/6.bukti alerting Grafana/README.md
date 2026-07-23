# Bukti Alerting Grafana

Simpan screenshot alerting Grafana di folder ini (minimal 3 alert rules).

Nama file yang direkomendasikan:
1. 1.rules_high_latency.jpg       — Rule alert untuk high latency
2. 2.notifikasi_high_latency.jpg   — Notifikasi alert high latency
3. 3.rules_error_rate.jpg          — Rule alert untuk error rate
4. 4.notifikasi_error_rate.jpg     — Notifikasi alert error rate
5. 5.rules_memory_usage.jpg        — Rule alert untuk memory usage
6. 6.notifikasi_memory_usage.jpg   — Notifikasi alert memory usage

## Cara membuat Alert Rules di Grafana:

### Alert 1: High Latency
- Condition: `prediction_latency_seconds > 1.0`
- Severity: Warning
- Message: "Prediction latency exceeded 1 second"

### Alert 2: High Error Rate
- Condition: `rate(prediction_error_total[5m]) > 0.1`
- Severity: Critical
- Message: "Error rate exceeded 10%"

### Alert 3: High Memory Usage
- Condition: `memory_usage_percent > 80`
- Severity: Warning
- Message: "Memory usage exceeded 80%"

## Langkah:
1. Buka Grafana → Alerting → Alert Rules
2. Klik "New Alert Rule"
3. Masukkan query PromQL sesuai condition di atas
4. Set threshold dan severity
5. Save dan screenshot rule yang sudah dibuat
6. Tunggu alert terpicu atau trigger manual, lalu screenshot notifikasi

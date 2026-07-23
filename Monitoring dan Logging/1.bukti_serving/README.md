# Bukti Serving

Simpan screenshot bukti serving model di folder ini.

Contoh nama file:
- serving_mlflow.jpg
- serving_docker.jpg

## Cara serve model:

### Opsi 1: MLflow Model Serve
```bash
mlflow models serve -m mlruns/0/<RUN_ID>/artifacts/model -p 5001 --no-conda
```

### Opsi 2: Docker Pull & Run (jika sudah push ke Docker Hub)
```bash
docker pull <username>/california-housing-model:latest
docker run -p 5001:8080 <username>/california-housing-model:latest
```

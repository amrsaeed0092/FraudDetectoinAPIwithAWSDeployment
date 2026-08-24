# Docker Deployment Guide

## 1. Build and Run Container
Execute the following command from the project root directory:

```bash
docker compose up --build
```

### Automated Actions Performed by Docker:
* **Builds** the FastAPI application image.
* **Installs** all required Python packages automatically.
* **Runs** the API using two high-performance worker processes.
* **Mounts** the `./model/artifacts` directory as a secure, read-only layer.
* **Exposes** the API externally at host port `8000`.

---

## 2. Access and API Testing

### Interactive Documentation
* **Open URL**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Core Endpoints Available
* **`GET /health`**: Checks application availability and system status.
* **`POST /api/v1/demo/fraud/score`**: Evaluates active transactions for fraud risks.

---

## 3. Container Management & Monitoring

### Check Container Health
Run this in a separate terminal window to verify service status and port mappings:
```bash
docker compose ps
```

### View Live Application Logs
Stream the last 50 log events directly from your API worker service:
```bash
docker compose logs api --tail 50
```

### Stop the Container Safely
Gracefully terminate operations and tear down internal networks:
```bash
docker compose down
```
## Push to DockerHub
### 1) Login to DockerHub
```bash
docker login
```

### 2) Find the local image id
```bash
docker images
```

### 3) Tag the Image for Docker Hub
* Format: docker tag LOCAL_IMAGE_NAME YOUR_USERNAME/REMOTE_REPO_NAME:TAG
```bash
docker tag fraud-detection-platform-api amrsaeed0092/fraud-detection-api:v1.0.0
```
* your_dockerhub_username: Your exact Docker Hub account name.
* fraud-detection-api: The name you want the repository to have on the web cloud.
* v1.0.0: The version tag (it is highly recommended to use a version number instead of leaving it as latest for tracking purposes).

### 4) Push the Image to the Cloud
```bash
docker push amrsaeed0092/fraud-detection-api:v1.0.0
```
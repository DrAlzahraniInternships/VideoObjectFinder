
# VideoObjectFinder

A Dockerized Python web app for **local video object & text detection** using:

- **OWL-ViT** (open-vocabulary object detection from Hugging Face)
- **EasyOCR** (text recognition on frames)

Upload a video (≤ 5 minutes), then search for objects (`person`, `laptop`, `glasses`) or text (`I HATE CODE`, `SALE`, etc.).  
The app extracts frames, runs detection, and returns a timestamped results table with clickable jump links.

---

## Prerequisites

Before you begin, ensure you have the following installed:

1. **Git** – [Install Git](https://git-scm.com/downloads)  
2. **Docker Desktop** – [Install Docker](https://docs.docker.com/get-docker/)  
3. **Linux/MacOS** – no extra setup needed  
4. **Windows** – install [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) and enable Docker’s WSL integration  

---

## Steps

### Step 1: Remove the existing code directory completely

```bash
rm -rf VideoObjectFinder
```
### Step 2: Clone the repository
```bash
git clone https://github.com/DrAlzahraniInternships/VideoObjectFinder.git
```
### Step 3: Navigate to the repository
```bash
cd VideoObjectFinder
```
### Step 4: Pull the latest version
```bash
git pull origin main
```
### Step 5: Build the Docker image
```bash
docker build -t video-object-finder .
```
### Step 6: Run the Docker container (with persistent caches)
```bash
docker run --rm -p 3001:3000 \
  -v "$PWD/jobs:/app/jobs" \
  -v "$PWD/.hfcache:/root/.cache/huggingface" \
  -v "$PWD/.eocache:/root/.EasyOCR" \
  --name video-object-finder video-object-finder
  ```
### Step 7: Access the app
- *Open your browser and go to:*
- 👉 http://localhost:3001

### Step 8: Stop and remove container/image
```bash
# stop the container (if still running in background)
docker rm -f video-object-finder
```
```bash
# remove image and clean up dangling cache/layers
docker rmi video-object-finder
docker system prune -f
```

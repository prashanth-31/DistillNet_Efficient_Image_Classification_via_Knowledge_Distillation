# Docker Deployment Guide for DistillNet

This guide explains how to deploy DistillNet using Docker and how to set up automated builds using GitHub Actions.

## Local Docker Deployment

### Using Docker Compose (Recommended)

The simplest way to run the application locally is with Docker Compose:

```bash
docker-compose up
```

This will build and start the Streamlit application with direct model loading.

### Manual Docker Deployment

Build and run the Streamlit Docker image:
```bash
docker build -t distillnet-streamlit -f Dockerfile.streamlit .
docker run -p 8501:8501 distillnet-streamlit
```

## Automated Builds with GitHub Actions

This project includes a GitHub Actions workflow that automatically builds and pushes Docker images to Docker Hub whenever changes are pushed to the main branch.

### Setup Instructions

1. **Create a Docker Hub Account**:
   - Sign up at [Docker Hub](https://hub.docker.com/) if you don't have an account

2. **Create a Docker Hub Access Token**:
   - Go to your Docker Hub account settings
   - Navigate to "Security" > "New Access Token"
   - Give it a name (e.g., "GitHub Actions")
   - Copy the generated token

3. **Add Secrets to GitHub Repository**:
   - Go to your GitHub repository
   - Navigate to "Settings" > "Secrets and variables" > "Actions"
   - Add the following secrets:
     - `DOCKERHUB_USERNAME`: Your Docker Hub username
     - `DOCKERHUB_TOKEN`: The access token you created

4. **Push to GitHub**:
   - The workflow will automatically trigger when you push to the main branch
   - You can also manually trigger it from the "Actions" tab in your GitHub repository

### Docker Images

The workflow builds and pushes two Docker images:

1. **distillnet-streamlit**: The Streamlit application for image classification
   - Tag: `your-username/distillnet-streamlit:latest`

2. **distillnet-training**: The training environment for model training
   - Tag: `your-username/distillnet-training:latest`

### Running the Docker Images

After the images are pushed to Docker Hub, you can run them with:

```bash
docker run -p 8501:8501 your-username/distillnet-streamlit:latest
```

Replace `your-username` with your Docker Hub username.

## Notes on Direct Model Loading

The Streamlit app loads models directly without using an API, which provides:

1. **Faster Inference**: No network overhead or serialization/deserialization
2. **Simpler Architecture**: No need to maintain a separate API service
3. **Lower Resource Usage**: Single container instead of multiple services 
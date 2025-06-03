#!/bin/bash

# Build the Docker image
echo "Building Docker image..."
docker build -t distillnet:test -f Dockerfile.streamlit .

# Run the container
echo "Running container..."
docker run --rm -d --name distillnet-test -p 8501:8501 distillnet:test

# Wait for the container to start
echo "Waiting for container to start..."
sleep 10

# Check the logs
echo "Container logs:"
docker logs distillnet-test

# Stop the container
echo "Stopping container..."
docker stop distillnet-test

echo "Test complete." 
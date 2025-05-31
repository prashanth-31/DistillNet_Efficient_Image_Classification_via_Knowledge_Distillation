# DistillNet Streamlit App

This directory contains the Streamlit application for interactive image classification using the trained student model.

## Running the App

To run the Streamlit app locally:

```bash
streamlit run app/streamlit_app.py
```

The app will be available at http://localhost:8501

## Features

- Upload an image for classification
- View the original image and the resized version used for model input
- Get the predicted class with confidence score
- See a bar chart of probabilities for all classes
- View model information and inference time

## Docker Deployment

You can also run the app using Docker:

```bash
docker build -t distillnet-streamlit -f Dockerfile.streamlit .
docker run -p 8501:8501 distillnet-streamlit
```

## Render Deployment

The app is configured for deployment on Render through the `render.yaml` file in the project root. 
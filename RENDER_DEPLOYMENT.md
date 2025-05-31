# Deploying DistillNet to Render

This guide explains how to deploy the DistillNet image classification system to Render.

## Prerequisites

1. A [Render account](https://render.com/)
2. Your project pushed to a Git repository (GitHub, GitLab, etc.)

## Deployment Steps

### 1. Connect Your Repository

1. Log in to your Render account
2. Go to the Dashboard and click "New" > "Blueprint"
3. Connect your Git repository containing the DistillNet project
4. Select the repository and click "Connect"

### 2. Configure Your Blueprint

Render will automatically detect the `render.yaml` file in your repository and set up the services:

- **distillnet-api**: FastAPI service for model inference
- **distillnet-streamlit**: Streamlit web application for interactive use
- **distillnet-training**: Background worker for model training (optional)

### 3. Deploy the Services

1. Review the configuration
2. Click "Apply" to start the deployment

### 4. Access Your Application

Once deployment is complete:

1. The Streamlit app will be available at: `https://distillnet-streamlit.onrender.com`
2. The API will be available at: `https://distillnet-api.onrender.com`

## Important Notes

- The free plan on Render has limited resources and may be slower for model inference
- Services on the free plan will spin down after periods of inactivity
- The first request after inactivity may take longer to process as the service spins up

## Troubleshooting

If you encounter issues:

1. Check the logs in the Render dashboard for each service
2. Ensure your models are properly saved and accessible in the deployed environment
3. Verify that all required dependencies are listed in `requirements.txt`

For more information, refer to the [Render documentation](https://render.com/docs). 
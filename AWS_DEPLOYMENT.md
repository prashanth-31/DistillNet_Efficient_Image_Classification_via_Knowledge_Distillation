# AWS Elastic Beanstalk Deployment Guide for DistillNet

This guide provides detailed instructions for deploying the DistillNet application to AWS Elastic Beanstalk, both manually and using the CI/CD pipeline with GitHub Actions.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Manual Deployment](#manual-deployment)
3. [CI/CD Deployment with GitHub Actions](#cicd-deployment-with-github-actions)
4. [Monitoring and Logging](#monitoring-and-logging)
5. [Scaling and Performance Optimization](#scaling-and-performance-optimization)
6. [Troubleshooting](#troubleshooting)

## Prerequisites

Before deploying to AWS Elastic Beanstalk, ensure you have the following:

- AWS account with appropriate permissions
- AWS CLI installed and configured
- Docker installed locally
- Git repository with the DistillNet code
- Trained model files in `app/models/` directory (or a fallback mechanism for pretrained models)

### Setting up AWS CLI

1. Install the AWS CLI:
   ```bash
   pip install awscli
   ```

2. Configure AWS CLI with your credentials:
   ```bash
   aws configure
   ```

3. Install the Elastic Beanstalk CLI:
   ```bash
   pip install awsebcli
   ```

## Manual Deployment

### Step 1: Create an ECR Repository

1. Create an Amazon ECR repository to store your Docker image:
   ```bash
   aws ecr create-repository --repository-name distillnet --region <your-region>
   ```

2. Log in to the ECR repository:
   ```bash
   aws ecr get-login-password --region <your-region> | docker login --username AWS --password-stdin <your-account-id>.dkr.ecr.<your-region>.amazonaws.com
   ```

### Step 2: Build and Push Docker Image

1. Build the Docker image:
   ```bash
   docker build -t distillnet -f Dockerfile.streamlit .
   ```

2. Tag the image for ECR:
   ```bash
   docker tag distillnet:latest <your-account-id>.dkr.ecr.<your-region>.amazonaws.com/distillnet:latest
   ```

3. Push the image to ECR:
   ```bash
   docker push <your-account-id>.dkr.ecr.<your-region>.amazonaws.com/distillnet:latest
   ```

### Step 3: Create Elastic Beanstalk Application Files

1. Create a `Dockerrun.aws.json` file:
   ```json
   {
     "AWSEBDockerrunVersion": "1",
     "Image": {
       "Name": "<your-account-id>.dkr.ecr.<your-region>.amazonaws.com/distillnet:latest",
       "Update": "true"
     },
     "Ports": [
       {
         "ContainerPort": "8501",
         "HostPort": "80"
       }
     ],
     "Logging": "/app/logs"
   }
   ```

2. Create the `.ebextensions` directory and configuration files:
   ```bash
   mkdir -p .ebextensions
   ```

3. Create Streamlit configuration in `.ebextensions/01_streamlit.config`:
   ```yaml
   option_settings:
     aws:elasticbeanstalk:application:environment:
       PYTHONPATH: "/app"
       STREAMLIT_SERVER_PORT: "8501"
       STREAMLIT_SERVER_ADDRESS: "0.0.0.0"
   ```

4. Create nginx configuration in `.ebextensions/02_nginx.config`:
   ```yaml
   files:
     "/etc/nginx/conf.d/proxy.conf":
       mode: "000644"
       owner: root
       group: root
       content: |
         upstream streamlit {
           server 127.0.0.1:8501;
         }
         
         server {
           listen 80;
         
           location / {
             proxy_pass http://streamlit;
             proxy_set_header X-Real-IP $remote_addr;
             proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
             proxy_set_header Host $http_host;
             proxy_set_header X-NginX-Proxy true;
             
             # Specific for websockets
             proxy_http_version 1.1;
             proxy_set_header Upgrade $http_upgrade;
             proxy_set_header Connection "upgrade";
             proxy_read_timeout 86400;
           }
         }
   
     "/opt/elasticbeanstalk/hooks/appdeploy/post/99_restart_nginx.sh":
       mode: "000755"
       owner: root
       group: root
       content: |
         #!/bin/bash
         service nginx restart
   ```

5. Create a deployment package:
   ```bash
   zip -r deploy.zip Dockerrun.aws.json .ebextensions
   ```

### Step 4: Deploy to Elastic Beanstalk

1. Initialize Elastic Beanstalk:
   ```bash
   eb init distillnet --region <your-region> --platform docker
   ```

2. Create an Elastic Beanstalk environment:
   ```bash
   eb create distillnet-env --cname distillnet --elb-type application --timeout 20
   ```

3. For subsequent deployments, use:
   ```bash
   eb deploy distillnet-env
   ```

4. Open the deployed application:
   ```bash
   eb open
   ```

## CI/CD Deployment with GitHub Actions

The repository includes a GitHub Actions workflow that automates the deployment process. When you push changes to the main branch, the workflow will:

1. Build the Docker image
2. Push it to Amazon ECR
3. Create a deployment package
4. Deploy to AWS Elastic Beanstalk

### Setting up GitHub Secrets

To enable the CI/CD pipeline, add the following secrets to your GitHub repository:

1. Go to your GitHub repository → Settings → Secrets and variables → Actions
2. Add the following repository secrets:

| Secret Name | Description |
|-------------|-------------|
| `AWS_ACCESS_KEY_ID` | Your AWS access key with permissions for ECR and Elastic Beanstalk |
| `AWS_SECRET_ACCESS_KEY` | Your AWS secret access key |
| `AWS_REGION` | The AWS region where you want to deploy (e.g., `us-east-1`) |

### Required IAM Permissions

Create an IAM user with the following permissions:

- AmazonECR-FullAccess
- AWSElasticBeanstalkFullAccess

For a more secure setup, create a custom policy with only the required permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "elasticbeanstalk:*",
        "ec2:*",
        "ecs:*",
        "ecr:*",
        "cloudwatch:*",
        "s3:*",
        "cloudformation:*",
        "autoscaling:*",
        "elasticloadbalancing:*",
        "iam:PassRole"
      ],
      "Resource": "*"
    }
  ]
}
```

## Monitoring and Logging

### CloudWatch Logs

The deployment automatically configures CloudWatch Logs for your application. You can view logs in the AWS Management Console:

1. Go to CloudWatch → Log groups
2. Find the log group for your Elastic Beanstalk environment

### Elastic Beanstalk Health Dashboard

1. Go to Elastic Beanstalk → Environments → distillnet-env
2. Click on "Health" in the navigation menu
3. Monitor the health of your environment and instances

## Scaling and Performance Optimization

### Environment Configuration

1. Go to Elastic Beanstalk → Environments → distillnet-env → Configuration
2. Under "Capacity", you can configure:
   - Instance type
   - Auto Scaling settings
   - Load balancer settings

### Recommended Settings

For production environments, consider the following settings:

- Instance Type: t3.small or larger
- Min instances: 2 (for high availability)
- Max instances: Based on your expected traffic
- Scaling trigger: CPU utilization > 70%

## Troubleshooting

### Common Issues

1. **Models not loading**: 
   - Check if the model files exist in the container
   - Verify the paths in the Dockerfile.streamlit
   - Consider using S3 to store and retrieve model files

2. **Application not accessible**:
   - Check security groups to ensure port 80 is open
   - Verify health check settings
   - Check the Elastic Beanstalk logs for errors

3. **Deployment fails**:
   - Check the deployment logs in GitHub Actions
   - Verify AWS credentials and permissions
   - Check the Elastic Beanstalk events for error messages

For more detailed troubleshooting, refer to [TROUBLESHOOTING.md](.github/workflows/TROUBLESHOOTING.md).

### Getting Help

If you encounter issues not covered in this guide:

1. Check the GitHub Actions workflow run logs
2. Review the AWS Elastic Beanstalk logs
3. Check CloudWatch Logs for container-level issues
4. Open an issue in the GitHub repository 
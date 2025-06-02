# Troubleshooting AWS Elastic Beanstalk Deployment

This guide provides solutions for common issues that might occur during the GitHub Actions deployment to AWS Elastic Beanstalk.

## Common Issues and Solutions

### 1. ECR Repository Does Not Exist

**Error message:**
```
An error occurred (RepositoryNotFoundException) when calling the DescribeRepositories operation: The repository with name 'distillnet' does not exist in the registry with id 'XXXXXXXXXXXX'
```

**Solution:**
The workflow is now configured to automatically create the ECR repository if it doesn't exist. If you still encounter this error:

1. Verify your IAM permissions include `ecr:CreateRepository`
2. Create the repository manually:
   ```bash
   aws ecr create-repository --repository-name distillnet --region <your-region>
   ```

### 2. Insufficient IAM Permissions

**Error message:**
```
User: arn:aws:iam::XXXXXXXXXXXX:user/github-actions is not authorized to perform: elasticbeanstalk:CreateApplication
```

**Solution:**
Ensure your IAM user has the required permissions. You can attach the following managed policies:
- `AmazonECR-FullAccess`
- `AWSElasticBeanstalkFullAccess`

For a more secure approach, create a custom policy with the minimum required permissions as described in the [AWS_DEPLOYMENT.md](../../AWS_DEPLOYMENT.md) file.

### 3. Elastic Beanstalk Environment Creation Fails

**Error message:**
```
ERROR: ServiceError - Create environment operation is complete, but with errors. For more information, see troubleshooting documentation.
```

**Solution:**
1. Check the Elastic Beanstalk console for detailed error messages
2. Common causes and fixes:
   - **VPC Configuration**: Ensure your VPC has both public and private subnets
   - **Security Groups**: Verify security groups allow traffic on ports 80 and 8501
   - **IAM Permissions**: The service role needs permissions to create resources
   - **Resource Limits**: You might have reached your AWS account limits

To get detailed error information:
```bash
aws elasticbeanstalk describe-events --environment-name distillnet-env --region <your-region>
```

### 4. Docker Build Fails

**Error message:**
```
failed to build: error building: failed to compute cache key: failed to calculate checksum of ref XXXXXXXXXXXX::XXXXXXXXXXXX: XXXXXXXXXXXX
```

**Solution:**
1. Check that your Dockerfile.streamlit is valid:
   ```bash
   docker build -t distillnet-test -f Dockerfile.streamlit .
   ```
2. Ensure all required files are present in the repository
3. Try clearing the GitHub Actions cache by adding a unique identifier to the workflow file
4. Check for disk space issues in the GitHub runner

### 5. Missing GitHub Secrets

**Error message:**
```
Error: Unable to locate credentials
```

**Solution:**
Verify that you've added all required secrets to your GitHub repository:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION`

To check if secrets are properly set (without revealing their values):
1. Go to your GitHub repository → Settings → Secrets and variables → Actions
2. Confirm all three secrets are listed

### 6. Model Files Not Found in Deployment

**Error message:**
```
Student model file not found. Using pretrained model instead.
```

**Solution:**
1. Ensure model files are included in your repository under `app/models/`
2. Verify the Dockerfile.streamlit correctly copies model files:
   ```dockerfile
   COPY app/models/*.pth /app/models/
   ```
3. For large model files, consider these alternatives:
   - Use AWS S3 to store models and download them during container startup
   - Add a script to download pretrained models if they don't exist
   - Use Git LFS for large file storage

### 7. Elastic Beanstalk Health Check Failures

**Error message:**
```
Health check failed: Elastic Load Balancer health check failure
```

**Solution:**
1. Verify your application is running on the correct port (8501 for Streamlit)
2. Check that the nginx configuration is correctly proxying requests
3. Ensure the health check path is accessible
4. Increase the health check timeout and threshold:
   ```yaml
   # Add to .ebextensions/04_healthcheck.config
   option_settings:
     aws:elasticbeanstalk:application:
       Application Healthcheck URL: /
     aws:elasticbeanstalk:environment:process:default:
       HealthCheckPath: /
       HealthCheckTimeout: 60
       HealthyThresholdCount: 2
       UnhealthyThresholdCount: 5
   ```

### 8. Deployment Timeout

**Error message:**
```
ERROR: Timed out while waiting for environment to reach state Ready
```

**Solution:**
1. Increase the deployment timeout in the GitHub workflow:
   ```yaml
   eb create distillnet-env --timeout 30
   ```
2. Check if the application is taking too long to start
3. Verify that your instance type has sufficient resources

### 9. WebSocket Connection Issues

**Error message:**
Users report that the Streamlit app keeps disconnecting or reloading.

**Solution:**
Ensure the nginx configuration correctly handles WebSocket connections:
```
proxy_http_version 1.1;
proxy_set_header Upgrade $http_upgrade;
proxy_set_header Connection "upgrade";
proxy_read_timeout 86400;
```

### 10. HTTPS/SSL Configuration

**Issue:**
Need to enable HTTPS for your application.

**Solution:**
1. Create an SSL certificate using AWS Certificate Manager (ACM)
2. Configure your Elastic Beanstalk environment to use HTTPS:
   ```yaml
   # Add to .ebextensions/05_https.config
   option_settings:
     aws:elb:listener:443:
       ListenerProtocol: HTTPS
       SSLCertificateId: arn:aws:acm:region:account-id:certificate/certificate-id
       InstancePort: 80
       InstanceProtocol: HTTP
   ```

## Getting Help

If you encounter issues not covered in this guide:

1. Check the GitHub Actions workflow run logs for detailed error messages
2. Review the AWS Elastic Beanstalk logs in the AWS Management Console:
   - Go to Elastic Beanstalk → Environments → distillnet-env → Logs
   - Request logs or enable log streaming
3. Check CloudWatch Logs for container-level issues
4. Use the AWS Elastic Beanstalk CLI for troubleshooting:
   ```bash
   eb logs
   eb health
   eb status
   ```
5. Open an issue in the GitHub repository with details about the error 
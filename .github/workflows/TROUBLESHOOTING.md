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

### 2. Docker Build Fails with COPY Command Error

**Error message:**
```
ERROR: failed to solve: failed to compute cache key: failed to calculate checksum of ref XXXXXXXXXXXX::XXXXXXXXXXXX: "/||": not found
```

**Solution:**
This error occurs when using shell redirection or logical operators in Docker COPY commands. Docker doesn't support shell features like `2>/dev/null` or `||` in COPY instructions.

1. Remove any shell redirection or logical operators from COPY commands in your Dockerfile
2. For conditional file copying, use multi-stage builds or handle the logic in a script that runs at container startup
3. If trying to copy model files that may not exist, make sure the directory structure exists first and handle missing files in the startup script

Example of correct approach:
```dockerfile
# Create the directory first
RUN mkdir -p /app/models

# Copy project files (models will be included if they exist)
COPY . .

# Handle missing models in startup script
RUN echo '#!/bin/bash\n\
if [ ! -f /app/models/model.pth ]; then\n\
  echo "Generating model files..."\n\
  # Generate or download models\n\
fi' > /app/startup.sh
```

### 3. Missing IAM Instance Profile Error

**Error message:**
```
ERROR   The instance profile aws-elasticbeanstalk-ec2-role associated with the environment does not exist.
ERROR   Failed to launch environment.
```

**Solution:**
Elastic Beanstalk requires specific IAM roles to function properly. When you see this error:

1. The workflow now automatically creates these roles if they don't exist:
   - `aws-elasticbeanstalk-ec2-role`: Instance profile for EC2 instances
   - `aws-elasticbeanstalk-service-role`: Service role for Elastic Beanstalk

2. If you're deploying manually or still encounter this error:
   ```bash
   # Create EC2 role and instance profile
   aws iam create-role --role-name aws-elasticbeanstalk-ec2-role --assume-role-policy-document file://ec2-trust-policy.json
   aws iam create-instance-profile --instance-profile-name aws-elasticbeanstalk-ec2-role
   aws iam add-role-to-instance-profile --instance-profile-name aws-elasticbeanstalk-ec2-role --role-name aws-elasticbeanstalk-ec2-role
   
   # Create service role
   aws iam create-role --role-name aws-elasticbeanstalk-service-role --assume-role-policy-document file://service-trust-policy.json
   ```

3. Ensure your IAM user has permissions to create and manage IAM roles
4. Specify these roles explicitly when creating your environment:
   ```bash
   eb create distillnet-env --service-role aws-elasticbeanstalk-service-role --instance-profile aws-elasticbeanstalk-ec2-role
   ```

### 4. Elastic Beanstalk CLI Argument Error

**Error message:**
```
ERROR: NotFoundError - Environment "distillnet-env" not Found.
Creating new environment...
usage: eb create <environment_name> [options ...]
eb: error: unrecognized arguments: --version-label distillnet-80e678a1
```

**Solution:**
The Elastic Beanstalk CLI can be very particular about argument formats and versions. When encountering argument errors:

1. Simplify your command by removing problematic arguments
2. Use only the essential arguments needed for deployment:
   ```bash
   eb create distillnet-env \
     --cname distillnet-short \
     --elb-type application \
     --instance-type t2.small \
     --platform docker
   ```
3. Avoid using `--version-label` or `--platform-version` if they cause errors
4. Check your EB CLI version with `eb --version` and consider updating if needed
5. For debugging, run `eb platform list` to see supported platforms

**Note:** Different versions of the EB CLI may accept different argument formats. The simplified command above should work with most versions.

### 5. Insufficient IAM Permissions

**Error message:**
```
User: arn:aws:iam::XXXXXXXXXXXX:user/github-actions is not authorized to perform: elasticbeanstalk:CreateApplication
```

**Solution:**
Ensure your IAM user has the required permissions. You can attach the following managed policies:
- `AmazonECR-FullAccess`
- `AWSElasticBeanstalkFullAccess`

For a more secure approach, create a custom policy with the minimum required permissions as described in the [AWS_DEPLOYMENT.md](../../AWS_DEPLOYMENT.md) file.

### 6. Elastic Beanstalk Environment Creation Fails

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

### 7. Docker Build Fails

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

### 8. Missing GitHub Secrets

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

### 9. Model Files Not Found in Deployment

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

### 10. Elastic Beanstalk Health Check Failures

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

### 11. Deployment Timeout

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

### 12. WebSocket Connection Issues

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

### 13. HTTPS/SSL Configuration

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
# DistillNet: Efficient Image Classification via Knowledge Distillation

DistillNet is a comprehensive MLOps pipeline for knowledge distillation in image classification. The project demonstrates how to train a smaller, efficient "student" model that learns from a larger "teacher" model while maintaining comparable performance.

## 📋 Project Overview

This project implements a complete MLOps pipeline for knowledge distillation with the following components:

- **Dataset**: CIFAR-10 image classification dataset
- **Teacher Model**: ResNet50 (high performance, but computationally expensive)
- **Student Model**: ResNet18 (smaller, more efficient, suitable for deployment)
- **Knowledge Distillation**: Transfer knowledge from teacher to student using KL divergence loss
- **MLOps Tools**: MLflow for experiment tracking and model versioning
- **Deployment**: 
  - Streamlit interactive application for image classification with direct model loading
  - Deployable via Docker and Render

## 🛠️ Project Structure

```
├── api/                  # API server for model deployment (optional)
│   ├── main.py           # FastAPI application
│   └── static/           # Static files for web UI
├── app/                  # Streamlit application
│   └── streamlit_app.py  # Streamlit app for interactive image classification
├── data/                 # Dataset storage (auto-downloaded)
├── models/               # Saved model weights
├── mlruns/               # MLflow experiment tracking data
├── notebooks/            # Jupyter notebooks for exploration
├── src/                  # Source code
│   ├── models/           # Model definitions
│   │   ├── __init__.py   # Model factory functions
│   │   └── student_model.py  # Student model architecture
│   ├── distillation_loss.py  # Knowledge distillation loss function
│   ├── evaluate.py       # Model evaluation script
│   ├── train_teacher.py  # Teacher model training script
│   ├── train_student.py  # Student model training script
│   └── utils.py          # Utility functions
├── .github/              # GitHub configuration
│   └── workflows/        # GitHub Actions workflows
├── config.yaml           # Configuration file
├── Dockerfile.streamlit  # Docker container definition for Streamlit app
├── Dockerfile.training   # Docker container definition for training
├── docker-compose.yml    # Docker Compose configuration for easy deployment
├── DOCKER_DEPLOYMENT.md  # Docker deployment guide
├── render.yaml           # Configuration for Render deployment
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- PyTorch and torchvision
- CUDA-capable GPU (optional, but recommended for training)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/distillnet.git
cd distillnet
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Training Pipeline

1. Train the teacher model:
```bash
python src/train_teacher.py
```

2. Train the student model with knowledge distillation:
```bash
python src/train_student.py
```

3. Evaluate and compare models:
```bash
python src/evaluate.py
```

### Model Deployment

#### Local Deployment

1. Start the Streamlit app (direct model loading, no API required):
```bash
streamlit run app/streamlit_app.py
```

2. Open your browser and navigate to `http://localhost:8501` to use the Streamlit interface.

#### Docker Deployment

For detailed instructions on Docker deployment and setting up GitHub Actions for automated builds, see [DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md).

#### Render Deployment

This project includes configuration for deployment on [Render](https://render.com/):

1. Push your code to a GitHub repository.

2. Log in to Render and create a new "Blueprint" instance.

3. Connect your GitHub repository and Render will automatically detect the `render.yaml` configuration.

4. Deploy the services defined in the configuration.

## 📊 MLflow Tracking

This project uses MLflow to track experiments. To view the MLflow UI:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Then open your browser and navigate to `http://localhost:5000`.

## 🧪 Knowledge Distillation

The knowledge distillation process involves:

1. Training a large teacher model (ResNet50) on CIFAR-10
2. Using the teacher's soft predictions to guide the training of a smaller student model (ResNet18)
3. Combining standard cross-entropy loss with KL divergence loss to transfer knowledge

The distillation loss is defined as:

```
L = α * KL(softmax(student_logits/T), softmax(teacher_logits/T)) + β * CrossEntropy(student_logits, labels)
```

where:
- T is the temperature parameter (higher values produce softer probability distributions)
- α and β are weights for the distillation and standard loss components

## 📈 Performance Comparison

| Metric | Teacher (ResNet50) | Student (ResNet18) |
|--------|-------------------|-------------------|
| Parameters | ~23.5M | ~11.2M |
| Accuracy | ~93-95% | ~92-93% |
| Inference Time | ~10-20ms | ~5-10ms |
| Model Size | ~90MB | ~44MB |

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

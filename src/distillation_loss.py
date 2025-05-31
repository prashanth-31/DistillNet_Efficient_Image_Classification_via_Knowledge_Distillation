import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss:
    """
    Knowledge Distillation Loss
    
    Combines KL divergence loss between teacher and student soft logits
    and standard cross-entropy loss between student predictions and true labels.
    
    Args:
        alpha (float): Weight for distillation loss (soft targets)
        beta (float): Weight for student loss (hard targets)
        temperature (float): Temperature for softening probability distributions
    """
    
    def __init__(self, alpha=0.5, beta=0.5, temperature=4.0):
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()
        
    def __call__(self, student_logits, teacher_logits, labels):
        """
        Calculate the distillation loss
        
        Args:
            student_logits (torch.Tensor): Logits from the student model
            teacher_logits (torch.Tensor): Logits from the teacher model
            labels (torch.Tensor): True labels
            
        Returns:
            torch.Tensor: Combined loss
        """
        # Standard cross-entropy loss between student predictions and true labels
        hard_loss = self.ce_loss(student_logits, labels)
        
        # KL divergence loss between teacher and student soft logits
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (self.temperature ** 2)
        
        # Combined loss
        return self.alpha * soft_loss + self.beta * hard_loss

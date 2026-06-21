import torch
import torch.nn as nn
import torch.nn.functional as F

class RKDLoss(nn.Module):
    """
    Relational Knowledge Distillation Engine.
    Computes Distance and Angle penalties between student and teacher representations.
    """
    def __init__(
        self,
        student_dim: int,
        teacher_dim: int,
        alpha: float = 1.0,
        beta: float = 0.5,
        alignment_strategy: str = "linear",
        num_triplets: int = 1000
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.alignment_strategy = alignment_strategy
        self.num_triplets = num_triplets
        
        # Auxiliary projection: map student to teacher dim for distance computation
        # Discarded at inference
        self.aux_proj = nn.Linear(student_dim, teacher_dim)
        
    def _align_sequences(self, teacher_acts: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        Aligns the teacher sequence length to the student sequence length.
        """
        B, T_teacher, D = teacher_acts.shape
        if T_teacher == target_len:
            return teacher_acts
            
        if self.alignment_strategy == "linear":
            # interpolate expects (B, Channels, Length)
            teacher_acts = teacher_acts.transpose(1, 2)
            aligned = F.interpolate(teacher_acts, size=target_len, mode='linear', align_corners=False)
            return aligned.transpose(1, 2)
        elif self.alignment_strategy == "nearest":
            # interpolate with nearest
            teacher_acts = teacher_acts.transpose(1, 2)
            aligned = F.interpolate(teacher_acts, size=target_len, mode='nearest')
            return aligned.transpose(1, 2)
        else:
            raise ValueError(f"Unknown alignment strategy: {self.alignment_strategy}")

    def _pdist(self, e: torch.Tensor) -> torch.Tensor:
        """Computes pairwise Euclidean distance matrix."""
        # e is (N, D)
        norm = (e ** 2).sum(dim=1, keepdim=True)
        dist = norm + norm.t() - 2 * torch.mm(e, e.t())
        # Add epsilon to prevent sqrt(0) -> NaN in gradient
        return torch.sqrt(torch.clamp(dist, min=1e-8))

    def _distance_loss(self, student_flat: torch.Tensor, teacher_flat: torch.Tensor) -> torch.Tensor:
        d_student = self._pdist(student_flat)
        d_teacher = self._pdist(teacher_flat)
        
        # Normalize by mean
        d_hat_student = d_student / (d_student.mean() + 1e-8)
        d_hat_teacher = d_teacher / (d_teacher.mean() + 1e-8)
        
        return F.huber_loss(d_hat_student, d_hat_teacher)

    def _angle_loss(self, student_flat: torch.Tensor, teacher_flat: torch.Tensor) -> torch.Tensor:
        N = student_flat.size(0)
        
        # Sample triplets uniformly
        # To avoid O(n^3) memory, sample indices
        idx_i = torch.randint(0, N, (self.num_triplets,), device=student_flat.device)
        idx_j = torch.randint(0, N, (self.num_triplets,), device=student_flat.device)
        idx_k = torch.randint(0, N, (self.num_triplets,), device=student_flat.device)
        
        def compute_angles(e):
            v1 = e[idx_i] - e[idx_j]
            v2 = e[idx_k] - e[idx_j]
            
            dot_product = (v1 * v2).sum(dim=1)
            norm_v1 = torch.norm(v1, dim=1)
            norm_v2 = torch.norm(v2, dim=1)
            
            eps = 1e-8 # Mandatory to prevent NaN
            cos_angle = dot_product / (norm_v1 * norm_v2 + eps)
            return cos_angle
            
        e_student = compute_angles(student_flat)
        e_teacher = compute_angles(teacher_flat)
        
        return F.huber_loss(e_student, e_teacher)

    def forward(
        self,
        student_activations: torch.Tensor, # (B, T_proj, D_student)
        teacher_activations: torch.Tensor, # (B, T_teacher, D_teacher)
    ) -> Dict[str, torch.Tensor]:
        
        B, T_proj, D_student = student_activations.shape
        
        # 1. Map student to teacher dim
        student_projected = self.aux_proj(student_activations)
        
        # 2. Align teacher temporal length to match student T_proj
        teacher_aligned = self._align_sequences(teacher_activations, T_proj)
        
        # Flatten batch and sequence dimensions for geometry computation
        # Geometry is independent of sequence order, it's about the point cloud
        student_flat = student_projected.reshape(-1, teacher_aligned.size(-1))
        teacher_flat = teacher_aligned.reshape(-1, teacher_aligned.size(-1))
        
        # Compute losses
        loss_dist = self._distance_loss(student_flat, teacher_flat)
        loss_angle = self._angle_loss(student_flat, teacher_flat)
        
        # Guard against NaNs
        if torch.isnan(loss_dist) or torch.isinf(loss_dist):
            raise RuntimeError("RKD Distance loss produced NaN/Inf!")
        if torch.isnan(loss_angle) or torch.isinf(loss_angle):
            raise RuntimeError("RKD Angle loss produced NaN/Inf!")
            
        return {
            "distance_loss": loss_dist,
            "angle_loss": loss_angle,
            "total_rkd_loss": self.alpha * loss_dist + self.beta * loss_angle
        }

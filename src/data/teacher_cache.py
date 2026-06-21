import os
import torch
from pathlib import Path
from typing import Optional

class TeacherCache:
    """
    Manages caching of teacher model's hooked intermediate activations to disk.
    This avoids running the huge teacher model during every epoch.
    """
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_path(self, sample_id: str) -> Path:
        # Sanitize sample_id for filename
        safe_id = "".join([c if c.isalnum() else "_" for c in sample_id])
        return self.cache_dir / f"{safe_id}.pt"
        
    def save(self, sample_id: str, activations: torch.Tensor):
        """
        Saves activations for a specific sample.
        Converts to float16 to save disk space as per PRD.
        """
        path = self._get_path(sample_id)
        # activations are saved as float16
        torch.save(activations.cpu().to(torch.float16), path)
        
    def load(self, sample_id: str, device: torch.device = torch.device('cpu')) -> Optional[torch.Tensor]:
        """
        Loads activations for a specific sample.
        Converts back to float32 for computation.
        """
        path = self._get_path(sample_id)
        if not path.exists():
            return None
            
        activations = torch.load(path, weights_only=True)
        return activations.to(device).to(torch.float32)
        
    def exists(self, sample_id: str) -> bool:
        return self._get_path(sample_id).exists()
        
    def load_batch(self, sample_ids: list[str], device: torch.device) -> Optional[torch.Tensor]:
        """
        Loads a batch of activations. Returns None if any sample is missing from cache.
        """
        batch_acts = []
        for sid in sample_ids:
            act = self.load(sid, device)
            if act is None:
                return None
            batch_acts.append(act)
            
        # Assuming all have same length for padding, or we pad them
        # Teacher sequences might have different lengths. For batching we might need padding.
        # But for RKD we align sequence length with student before flattening.
        # So padding here with 0s and keeping an attention mask would be required, 
        # or we just assume they are padded to max length in the dataset loader.
        
        # Simplified: pad sequence length to max in this batch
        max_len = max(a.size(0) for a in batch_acts)
        padded_acts = []
        for a in batch_acts:
            pad_len = max_len - a.size(0)
            if pad_len > 0:
                padded = torch.nn.functional.pad(a, (0, 0, 0, pad_len))
                padded_acts.append(padded)
            else:
                padded_acts.append(a)
                
        return torch.stack(padded_acts) # (B, T_teacher_max, D_teacher)

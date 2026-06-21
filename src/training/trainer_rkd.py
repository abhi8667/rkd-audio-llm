import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from rich.console import Console

from src.training.hooks import HookManager
from src.training.rkd_loss import RKDLoss
from src.data.teacher_cache import TeacherCache

console = Console()

class RKDTrainer:
    def __init__(
        self, 
        student_model, 
        teacher_model, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        config: dict
    ):
        self.student = student_model
        self.teacher = teacher_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.student.to(self.device)
        if self.teacher is not None:
            self.teacher.to(self.device)
            self.teacher.eval() # Teacher is always frozen
            
        # Optimizer includes RKD auxiliary projection parameters if added
        trainable_params = [p for p in self.student.parameters() if p.requires_grad]
        
        # Setup RKD Loss
        self.rkd_loss_fn = RKDLoss(
            student_dim=config.get('student_hook_dim', 1536), # example values
            teacher_dim=config.get('teacher_hook_dim', 4096),
            alpha=config.get('loss_weights', {}).get('alpha', 1.0),
            beta=config.get('loss_weights', {}).get('beta', 0.5),
            alignment_strategy=config.get('alignment_strategy', 'linear')
        ).to(self.device)
        
        trainable_params.extend(self.rkd_loss_fn.parameters())
        self.optimizer = AdamW(trainable_params, lr=config.get('learning_rate', 3e-4))
        
        # Setup Hooks
        self.hook_manager = HookManager()
        # In practice, you'd find the exact layers to hook based on the model architectures
        # e.g., self.hook_manager.register_student_hook(self.student.projection)
        
        # Setup Cache
        self.teacher_cache = TeacherCache(config.get('cache_dir', './cache/teacher_acts'))

    def train_epoch(self):
        self.student.train()
        total_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc="RKD Training")
        for batch in progress_bar:
            self.optimizer.zero_grad()
            self.hook_manager.clear_activations()
            
            waveforms = batch['waveforms'].to(self.device)
            attention_masks = batch['attention_masks'].to(self.device)
            sample_ids = batch['sample_id']
            
            # Text inputs setup (simplified)
            prompts = ["Intent: " for _ in range(len(waveforms))]
            targets = [p + intent for p, intent in zip(prompts, batch['intent_labels'])]
            tokenizer = self.student.tokenizer
            text_inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
            labels = tokenizer(targets, return_tensors="pt", padding=True).input_ids.to(self.device)
            
            # Forward student
            student_outputs = self.student(
                waveforms=waveforms,
                attention_masks=attention_masks,
                text_inputs=text_inputs,
                labels=labels
            )
            l_ce = student_outputs.loss
            
            # Extract student hooked activations
            # Note: actual name depends on where the hook was registered
            student_acts = self.hook_manager.get_activation("student")
            
            # Get teacher activations (from cache or live forward pass)
            teacher_acts = self.teacher_cache.load_batch(sample_ids, self.device)
            if teacher_acts is None:
                if self.teacher is None:
                    raise RuntimeError("Teacher activations not cached and live teacher model not loaded.")
                with torch.no_grad():
                    # Simplified teacher forward pass
                    self.teacher(waveforms, attention_masks)
                teacher_acts = self.hook_manager.get_activation("teacher")
                # Save to cache for next time
                for i, sid in enumerate(sample_ids):
                    self.teacher_cache.save(sid, teacher_acts[i])
                    
            # Compute RKD Loss
            rkd_losses = self.rkd_loss_fn(student_acts, teacher_acts)
            
            # Combined loss
            l_total = l_ce + rkd_losses["total_rkd_loss"]
            
            # Backward pass
            l_total.backward()
            self.optimizer.step()
            
            total_loss += l_total.item()
            progress_bar.set_postfix({
                'L_CE': l_ce.item(), 
                'L_Dist': rkd_losses["distance_loss"].item(), 
                'L_Ang': rkd_losses["angle_loss"].item()
            })
            
        return total_loss / len(self.train_loader)
        
    def cleanup(self):
        self.hook_manager.remove_hooks()

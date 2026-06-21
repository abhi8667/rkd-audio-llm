import torch
import torch.nn as nn
from typing import Dict, Any

class HookManager:
    """
    Manages forward hooks for extracting intermediate activations 
    from teacher and student models.
    """
    def __init__(self):
        self.activations: Dict[str, torch.Tensor] = {}
        self.hooks = []
        
    def _make_hook(self, name: str, detach: bool = False):
        def hook(module: nn.Module, input: Any, output: Any):
            # If output is a tuple (like from transformers), take the first element (hidden states)
            hidden_states = output[0] if isinstance(output, tuple) else output
            
            if detach:
                self.activations[name] = hidden_states.detach()
            else:
                self.activations[name] = hidden_states
        return hook

    def register_teacher_hook(self, module: nn.Module, name: str = "teacher"):
        """Registers a hook on the teacher. Must detach output."""
        hook_handle = module.register_forward_hook(self._make_hook(name, detach=True))
        self.hooks.append(hook_handle)
        
    def register_student_hook(self, module: nn.Module, name: str = "student"):
        """Registers a hook on the student. Does NOT detach output to allow backprop."""
        hook_handle = module.register_forward_hook(self._make_hook(name, detach=False))
        self.hooks.append(hook_handle)
        
    def get_activation(self, name: str) -> torch.Tensor:
        if name not in self.activations:
            raise ValueError(f"Activation '{name}' not found. Did the hook fire?")
        return self.activations[name]
        
    def clear_activations(self):
        """Clears memory to prevent leaks across batches."""
        self.activations.clear()
        
    def remove_hooks(self):
        """Removes all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

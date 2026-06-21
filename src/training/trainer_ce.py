import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from rich.console import Console

console = Console()

class CETrainer:
    def __init__(self, model, train_loader: DataLoader, val_loader: DataLoader, config: dict):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Only optimize parameters that require gradients (Projection Bridge + LoRA)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = AdamW(trainable_params, lr=config.get('learning_rate', 3e-4))
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        for batch in progress_bar:
            self.optimizer.zero_grad()
            
            waveforms = batch['waveforms'].to(self.device)
            attention_masks = batch['attention_masks'].to(self.device)
            
            # Simple prompt tokenization logic 
            # In practice, this needs proper padding and attention masks
            prompts = ["Intent: " for _ in range(len(waveforms))]
            targets = [p + intent for p, intent in zip(prompts, batch['intent_labels'])]
            
            # Tokenize targets
            # This is a simplified placeholder. A robust implementation would use tokenizer padding.
            tokenizer = self.model.tokenizer
            text_inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
            labels = tokenizer(targets, return_tensors="pt", padding=True).input_ids.to(self.device)
            
            # Forward pass
            outputs = self.model(
                waveforms=waveforms,
                attention_masks=attention_masks,
                text_inputs=text_inputs,
                labels=labels
            )
            
            # Cross-Entropy loss is automatically computed by causal LM if labels are passed
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                waveforms = batch['waveforms'].to(self.device)
                attention_masks = batch['attention_masks'].to(self.device)
                
                # Simplified tokenization for eval
                prompts = ["Intent: " for _ in range(len(waveforms))]
                targets = [p + intent for p, intent in zip(prompts, batch['intent_labels'])]
                
                tokenizer = self.model.tokenizer
                text_inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
                labels = tokenizer(targets, return_tensors="pt", padding=True).input_ids.to(self.device)
                
                outputs = self.model(
                    waveforms=waveforms,
                    attention_masks=attention_masks,
                    text_inputs=text_inputs,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
                
        return total_loss / len(self.val_loader)

    def train(self, num_epochs: int):
        for epoch in range(num_epochs):
            console.print(f"[bold blue]Epoch {epoch+1}/{num_epochs}[/]")
            train_loss = self.train_epoch()
            val_loss = self.validate()
            console.print(f"[green]Train Loss:[/] {train_loss:.4f} | [green]Val Loss:[/] {val_loss:.4f}")

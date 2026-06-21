import os
import json
import torch
import numpy as np
import soundfile as sf
from torch.utils.data import DataLoader
from rich.console import Console

from src.models.projection import ProjectionBridge
from src.models.audio_llm import AudioLLM
from src.data.dataset_loader import AudioLLMDataset, collate_fn
from src.training.trainer_ce import CETrainer

console = Console()

def create_dummy_data(data_dir="./data"):
    """Creates dummy audio files and a manifest.jsonl for testing."""
    os.makedirs(data_dir, exist_ok=True)
    manifest_path = os.path.join(data_dir, "manifest.jsonl")
    
    # Generate a couple of dummy wav files
    console.print("[yellow]Generating dummy audio data...[/]")
    samples = []
    for i in range(4):
        audio_path = os.path.join(data_dir, f"sample_{i}.wav")
        # 3 seconds of dummy audio at 16kHz
        audio = np.random.randn(16000 * 3).astype(np.float32) * 0.1
        sf.write(audio_path, audio, 16000)
        
        samples.append({
            "sample_id": f"dummy_{i}",
            "audio_path": audio_path,
            "transcript": "hello world",
            "intent_label": "greeting",
            "permits_model_training": True
        })
        
    with open(manifest_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
            
    return manifest_path

def main():
    console.print("[bold green]Starting RKD Audio-LLM Demo[/]")
    
    # 1. Prepare Data
    manifest_path = create_dummy_data()
    dataset = AudioLLMDataset(manifest_path)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    
    # 2. Setup Model
    console.print("[yellow]Initializing Model (this will download model weights if not cached)...[/]")
    # We use a very small decoder for the demo so it runs fast
    encoder_dim = 1024 # WavLM Large
    decoder_dim = 1536 # Example Qwen2.5-1.5B dim
    
    projection = ProjectionBridge(encoder_dim=encoder_dim, decoder_dim=decoder_dim, downsample_factor=2)
    
    model = AudioLLM(
        encoder_name="microsoft/wavlm-large", 
        decoder_name="Qwen/Qwen2.5-1.5B", 
        projection_module=projection,
        use_lora=True
    )
    
    # 3. Setup Trainer
    config = {"learning_rate": 3e-4}
    trainer = CETrainer(model=model, train_loader=dataloader, val_loader=dataloader, config=config)
    
    # 4. Train
    console.print("[bold blue]Starting Dummy Training Loop...[/]")
    trainer.train(num_epochs=1)
    
    console.print("[bold green]Demo Completed Successfully![/]")

if __name__ == "__main__":
    main()

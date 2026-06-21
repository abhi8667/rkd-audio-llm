import json
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple

class AudioLLMDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        sample_rate: int = 16000,
        max_audio_length_sec: int = 30,
        noise_library_path: Optional[str] = None
    ):
        """
        Args:
            manifest_path: Path to the JSONL dataset manifest.
            sample_rate: Target sample rate for audio.
            max_audio_length_sec: Maximum audio length in seconds. Audio will be chunked or truncated.
            noise_library_path: Optional path to a directory of noise profiles for on-the-fly augmentation.
        """
        self.manifest_path = manifest_path
        self.sample_rate = sample_rate
        self.max_audio_samples = sample_rate * max_audio_length_sec
        self.samples = self._load_manifest(manifest_path)
        
        # In a real implementation, we would load the noise library here
        self.noise_library = None 
        
    def _load_manifest(self, path: str) -> List[Dict]:
        samples = []
        with open(path, 'r') as f:
            for line in f:
                samples.append(json.loads(line.strip()))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample_info = self.samples[idx]
        
        # Ensure we don't ingest samples without permission
        if not sample_info.get("permits_model_training", True): # defaulting to True for placeholder
             raise ValueError(f"Sample {sample_info['sample_id']} does not permit model training.")

        audio_path = sample_info["audio_path"]
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Convert to mono if necessary
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Apply noise augmentation (placeholder for actual implementation)
        # waveform = self._inject_noise(waveform)
            
        # Truncate to max length
        if waveform.shape[1] > self.max_audio_samples:
            waveform = waveform[:, :self.max_audio_samples]

        return {
            "sample_id": sample_info.get("sample_id", str(idx)),
            "waveform": waveform.squeeze(0), # Remove channel dim for batching
            "transcript": sample_info.get("transcript", ""),
            "intent_label": sample_info.get("intent_label", None),
            "summary_label": sample_info.get("summary_label", None),
        }

def collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function to pad waveforms to the maximum length in the batch.
    """
    waveforms = [item["waveform"] for item in batch]
    transcripts = [item["transcript"] for item in batch]
    intent_labels = [item["intent_label"] for item in batch]
    summary_labels = [item["summary_label"] for item in batch]
    
    # Pad waveforms
    max_len = max([w.shape[0] for w in waveforms])
    
    padded_waveforms = torch.zeros(len(waveforms), max_len)
    attention_masks = torch.zeros(len(waveforms), max_len, dtype=torch.bool)
    
    for i, w in enumerate(waveforms):
        length = w.shape[0]
        padded_waveforms[i, :length] = w
        attention_masks[i, :length] = True
        
    return {
        "waveforms": padded_waveforms,
        "attention_masks": attention_masks,
        "transcripts": transcripts,
        "intent_labels": intent_labels,
        "summary_labels": summary_labels
    }

def get_dataloader(
    manifest_path: str,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    **dataset_kwargs
) -> DataLoader:
    dataset = AudioLLMDataset(manifest_path, **dataset_kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

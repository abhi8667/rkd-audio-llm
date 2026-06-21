import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from typing import Optional, Dict

class AudioLLM(nn.Module):
    """
    End-to-End Multimodal Audio-LLM.
    Wraps the frozen audio encoder, projection bridge, and LoRA-adapted decoder.
    """
    def __init__(
        self,
        encoder_name: str = "microsoft/wavlm-large",
        decoder_name: str = "Qwen/Qwen2.5-1.5B",
        projection_module: nn.Module = None,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32
    ):
        super().__init__()
        
        # 1. Audio Encoder (Frozen)
        self.encoder = AutoModel.from_pretrained(encoder_name)
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        # 2. Projection Bridge (Trainable)
        self.projection = projection_module
        
        # 3. Compact Decoder
        self.decoder = AutoModelForCausalLM.from_pretrained(
            decoder_name,
            torch_dtype=torch.float16,
            device_map="auto" # or just load to CPU and let trainer manage
        )
        self.tokenizer = AutoTokenizer.from_pretrained(decoder_name)
        
        # Freeze decoder base weights
        for param in self.decoder.parameters():
            param.requires_grad = False
            
        # Attach LoRA adapters
        if use_lora:
            peft_config = LoraConfig(
                task_type="CAUSAL_LM",
                inference_mode=False,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=0.1,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            )
            self.decoder = get_peft_model(self.decoder, peft_config)
            
    def forward(
        self,
        waveforms: torch.Tensor,
        attention_masks: torch.Tensor,
        text_inputs: Optional[Dict[str, torch.Tensor]] = None,
        labels: Optional[torch.Tensor] = None
    ):
        """
        Args:
            waveforms: (B, T) tensor
            attention_masks: (B, T) tensor for audio
            text_inputs: Tokenized prompt text
            labels: Tokenized targets for CE loss
        """
        # 1. Encode audio
        encoder_outputs = self.encoder(
            input_values=waveforms,
            attention_mask=attention_masks
        )
        encoder_states = encoder_outputs.last_hidden_state # (B, T_audio, D_enc)
        
        # 2. Project to decoder dimension
        projected_embeds = self.projection(encoder_states) # (B, T_proj, D_dec)
        
        # 3. Combine with text prompts if provided
        if text_inputs is not None:
            text_embeds = self.decoder.model.embed_tokens(text_inputs['input_ids'])
            # Concat audio embeddings as a prefix
            inputs_embeds = torch.cat([projected_embeds, text_embeds], dim=1)
            
            # Combine attention masks
            # Note: T_proj length might vary depending on downsampling, need valid mask
            audio_mask = torch.ones(
                (projected_embeds.size(0), projected_embeds.size(1)),
                dtype=torch.bool, device=projected_embeds.device
            )
            combined_attention_mask = torch.cat([audio_mask, text_inputs['attention_mask']], dim=1)
            
            # Pad labels with -100 for audio prefix so loss isn't computed on audio embeddings
            if labels is not None:
                audio_labels = torch.full(
                    (labels.size(0), projected_embeds.size(1)),
                    -100, dtype=labels.dtype, device=labels.device
                )
                labels = torch.cat([audio_labels, labels], dim=1)
                
        else:
            inputs_embeds = projected_embeds
            combined_attention_mask = torch.ones(
                (projected_embeds.size(0), projected_embeds.size(1)),
                dtype=torch.bool, device=projected_embeds.device
            )
            
        # 4. Decoder forward pass
        outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_attention_mask,
            labels=labels
        )
        
        return outputs

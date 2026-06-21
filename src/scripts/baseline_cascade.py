import argparse
import time
import torch
import torchaudio
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from rich.console import Console

console = Console()

class BaselineCascade:
    def __init__(self, asr_model_name: str = "openai/whisper-small", llm_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        console.print(f"[bold blue]Initializing ASR Model:[/] {asr_model_name}")
        self.asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=asr_model_name,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        console.print(f"[bold blue]Initializing LLM:[/] {llm_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model_name, 
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
    def process_audio(self, audio_path: str, prompt_template: str) -> dict:
        """
        Runs the cascaded pipeline: ASR -> LLM
        """
        start_time = time.time()
        
        # Step 1: ASR
        console.print("[yellow]Running ASR...[/]")
        asr_start = time.time()
        asr_result = self.asr_pipeline(audio_path)
        transcript = asr_result["text"]
        asr_latency = time.time() - asr_start
        console.print(f"[green]Transcript:[/] {transcript}")
        
        # Step 2: LLM
        console.print("[yellow]Running LLM...[/]")
        prompt = prompt_template.format(transcript=transcript)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.llm.device)
        
        llm_start = time.time()
        with torch.no_grad():
            outputs = self.llm.generate(**inputs, max_new_tokens=128)
        
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        llm_latency = time.time() - llm_start
        
        total_latency = time.time() - start_time
        
        return {
            "transcript": transcript,
            "response": response,
            "asr_latency_sec": asr_latency,
            "llm_latency_sec": llm_latency,
            "total_latency_sec": total_latency
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the cascaded ASR -> LLM baseline.")
    parser.add_argument("--audio", type=str, required=True, help="Path to the audio file.")
    parser.add_argument("--task", type=str, choices=["intent", "summarize"], default="intent", help="Task to perform.")
    args = parser.parse_args()
    
    cascade = BaselineCascade()
    
    if args.task == "intent":
        prompt_template = "Classify the intent of the following transcript. Options: [complaint, inquiry, greeting, other].\nTranscript: {transcript}\nIntent:"
    else:
        prompt_template = "Summarize the following transcript:\n{transcript}\nSummary:"
        
    result = cascade.process_audio(args.audio, prompt_template)
    
    console.print("\n[bold magenta]Results:[/]")
    console.print(f"Final Response: {result['response']}")
    console.print(f"Total Latency: {result['total_latency_sec']:.2f}s (ASR: {result['asr_latency_sec']:.2f}s, LLM: {result['llm_latency_sec']:.2f}s)")

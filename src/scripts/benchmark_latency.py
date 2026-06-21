import argparse
import time
import torch
import numpy as np
from rich.console import Console
from rich.table import Table

from src.scripts.infer import AudioLLM
from src.scripts.baseline_cascade import BaselineCascade

console = Console()

def generate_dummy_audio(duration_sec: int, sample_rate: int = 16000) -> np.ndarray:
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec))
    # A simple sine wave + noise
    audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t))
    return audio.astype(np.float32)

def benchmark_model(model_func, audio: np.ndarray, iterations: int = 10):
    latencies = []
    
    # Warmup
    console.print("Warming up...")
    _ = model_func(audio)
    
    # Benchmark
    console.print(f"Running {iterations} iterations...")
    for _ in range(iterations):
        start = time.time()
        _ = model_func(audio)
        latencies.append(time.time() - start)
        
    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    
    return avg_latency, p95_latency

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, help="Path to RKD student checkpoint", required=True)
    parser.add_argument("--duration", type=int, default=5, help="Audio duration in seconds to benchmark")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations")
    args = parser.parse_args()
    
    audio = generate_dummy_audio(args.duration)
    
    # Init models
    console.print("[bold]Initializing Models...[/bold]")
    rkd_model = AudioLLM(checkpoint_path=args.checkpoint)
    cascade_model = BaselineCascade()
    
    # Benchmark RKD Student
    console.print(f"\n[bold green]Benchmarking RKD Student Model (duration={args.duration}s)[/]")
    def run_rkd(a):
        return rkd_model.generate(a, sample_rate=16000, prompt="Intent:")
    rkd_avg, rkd_p95 = benchmark_model(run_rkd, audio, args.iterations)
    
    # Benchmark Cascade
    console.print(f"\n[bold yellow]Benchmarking Cascaded Baseline (duration={args.duration}s)[/]")
    # For dummy audio, ASR will produce garbage but it will still compute
    import tempfile
    import soundfile as sf
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio, 16000)
        def run_cascade(a):
            return cascade_model.process_audio(tmp.name, "Intent: {transcript}")
        cascade_avg, cascade_p95 = benchmark_model(run_cascade, audio, args.iterations)
    
    speedup = cascade_avg / rkd_avg
    
    # Reporting
    table = Table(title="Latency Benchmark Results")
    table.add_column("Model", style="cyan")
    table.add_column("Avg Latency (s)", style="magenta")
    table.add_column("P95 Latency (s)", style="magenta")
    table.add_column("Speedup vs Cascade", style="green")
    
    table.add_row("Cascaded ASR+LLM", f"{cascade_avg:.3f}", f"{cascade_p95:.3f}", "1.0x")
    table.add_row("RKD End-to-End", f"{rkd_avg:.3f}", f"{rkd_p95:.3f}", f"{speedup:.2f}x")
    
    console.print(table)
    
    if speedup >= 2.0:
        console.print("[bold green]Success: Target 2x speedup achieved![/]")
    else:
        console.print("[bold red]Warning: Target 2x speedup not achieved.[/]")

if __name__ == "__main__":
    main()

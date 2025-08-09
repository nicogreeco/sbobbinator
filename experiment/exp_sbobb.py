import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import time
import gc
import pandas as pd
import json
import subprocess
from faster_whisper import WhisperModel, BatchedInferencePipeline

def get_gpu_memory_usage():
    """Return GPU memory usage in MB using nvidia-smi"""
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        # If multiple GPUs, return the memory of the first one
        return float(result.strip().split('\n')[0])
    except (subprocess.SubprocessError, FileNotFoundError, ValueError):
        print("Warning: Could not get GPU memory usage")
        return 0

def benchmark_transcription(
    audio_file_path: str,
    language_code: str,
    batch_sizes: list[int],
    model_names: list[str],
    quants: list[str]
) -> pd.DataFrame:
    """
    For each Whisper model in `model_names`:
      1. Load the model (on CPU/GPU).
      2. For each batch_size in `batch_sizes`, call `model_batch.transcribe(...)`
         and measure only that transcription time.
      3. Delete the model & pipeline from memory before loading the next model.

    Returns a DataFrame with columns: [model, batch_size, duration_s].
    """
    results = []
    transcripts = []

    for model_name in model_names:
        for quant in quants:
            try:
                # Measure memory before model load
                memory_before = get_gpu_memory_usage()
                
                # Load WhisperModel
                print(f"\nLoading model '{model_name} - {quant}'...")
                model = WhisperModel(model_name, device="auto", compute_type=quant)
                model_batch = BatchedInferencePipeline(model=model)
                
                # Measure memory after model load
                memory_after = get_gpu_memory_usage()
                model_memory = memory_after - memory_before
                print(f"Model memory usage: {model_memory:.2f} MB")

                for bs in batch_sizes:
                    try:
                        print(f"  → Benchmarking batch_size={bs} ... \n", end="", flush=True)
                        start = time.time()
                        # Note: we convert segments to a list so that transcription actually runs now.
                        segments, _ = model_batch.transcribe(
                            audio_file_path,
                            batch_size=bs,
                            language=language_code,
                            log_progress=True,
                            word_timestamps=False
                        )
                        segments = list(segments)
                        duration = time.time() - start
                        result = [segment.text for segment in segments]
                        
                        results.append({
                            "model": model_name,
                            "batch_size": bs,
                            "comp_type": quant,
                            "duration_s": duration,
                            "memory_mb": model_memory
                        })

                        transcripts.append({
                            "model": model_name,
                            "batch_size": bs,
                            "comp_type": quant,
                            "transcript": result
                        })

                        print(f"  done ({duration:.2f} s)")
                    except RuntimeError as e:
                        if "CUDA" in str(e) and "out of memory" in str(e):
                            print(f"  → CUDA out of memory for batch_size={bs}. Skipping.")
                            # Record the failure in results
                            results.append({
                                "model": model_name,
                                "batch_size": bs,
                                "comp_type": quant,
                                "duration_s": None,
                                "memory_mb": model_memory,
                                "error": "CUDA OOM"
                            })
                        else:
                            # Re-raise other errors
                            raise e

                # Clean up before loading next model
                del model_batch
                del model
                gc.collect()
                # Give some time for GPU memory to be released
                time.sleep(2)
                
            except RuntimeError as e:
                if "CUDA" in str(e) and "out of memory" in str(e):
                    print(f"  → CUDA out of memory for model {model_name} - {quant}. Skipping.")
                    # Record the failure in results
                    for bs in batch_sizes:
                        results.append({
                            "model": model_name,
                            "batch_size": bs,
                            "comp_type": quant,
                            "duration_s": None,
                            "memory_mb": None,
                            "error": "CUDA OOM at model load"
                        })
                else:
                    # Re-raise other errors
                    raise e

    df = pd.DataFrame(results)
    # (Optional) sort so that all rows for each model are together:
    df = df.sort_values(by=["model", "batch_size", "comp_type"]).reset_index(drop=True)
    return df, transcripts

audio_path = r"d:\Cartelle Utente\Download\Attention in transformers, step-by-step Deep Learning Chapter 6.mp3"
lang = "en"
batches = [2, 4, 8, 12]
quants = ['int8', 'float16', 'float32']
models = ["medium", "large-v3", "distil-large-v3"]

# Give a clean start
time.sleep(2)  # Wait a bit to allow previous processes to release GPU memory

df_timings, transcripts = benchmark_transcription(
    audio_file_path=audio_path,
    language_code=lang,
    batch_sizes=batches,
    model_names=models,
    quants=quants
)

df_timings.to_csv("results_grid_gpu.csv", index=False)
with open('transcripts_results_gpu.json', "w", encoding="utf-8") as f:
    json.dump(transcripts, f, indent=4)
    
print("\nBenchmark results:")
print(df_timings.to_markdown(index=False))
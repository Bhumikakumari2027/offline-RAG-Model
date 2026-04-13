# hardware_check.py
import os
import psutil
import platform
import torch
import subprocess

def get_system_info():
    info = {}

    # --- CPU Name ---
    cpu_name = platform.processor() or platform.machine()
    try:
        if os.name == 'nt':
            # Use WMIC to get full CPU name on Windows
            out = subprocess.check_output(['wmic', 'cpu', 'get', 'Name'], stderr=subprocess.STDOUT, text=True)
            lines = [ln.strip() for ln in out.splitlines() if ln.strip() and 'Name' not in ln]
            if lines:
                cpu_name = lines[0]
    except Exception:
        pass
    info['cpu'] = cpu_name

    # --- CPU Max Frequency and Usage ---
    try:
        freq = psutil.cpu_freq()
        info['cpu_max_ghz'] = round((freq.max or 0.0) / 1000.0, 2) if freq else None
        info['cpu_usage_percent'] = psutil.cpu_percent(interval=1)
    except Exception:
        info['cpu_max_ghz'] = None
        info['cpu_usage_percent'] = None

    # --- RAM ---
    mem = psutil.virtual_memory()
    info['ram_gb'] = round(mem.total / (1024**3), 1)
    info['ram_usage_percent'] = mem.percent

    # --- GPU ---
    info['cuda_available'] = torch.cuda.is_available()
    if info['cuda_available']:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_props = torch.cuda.get_device_properties(0)
        info['gpu_name'] = gpu_name
        info['gpu_vram_gb'] = round(gpu_props.total_memory / (1024**3), 1)
        try:
            # Current GPU usage % (requires torch 2.1+ or nvidia-smi fallback)
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            info['gpu_usage_percent'] = util.gpu
            info['gpu_mem_usage_percent'] = util.memory
            pynvml.nvmlShutdown()
        except Exception:
            info['gpu_usage_percent'] = None
            info['gpu_mem_usage_percent'] = None
        info['gpu_type'] = "nvidia" if "nvidia" in gpu_name.lower() else "non-nvidia"
    else:
        info['gpu_name'] = "N/A"
        info['gpu_vram_gb'] = 0
        info['gpu_usage_percent'] = 0
        info['gpu_mem_usage_percent'] = 0
        info['gpu_type'] = "none"

    return info


def choose_preset(sysinfo=None):
    """Decides performance preset and configuration based on system info."""
    if sysinfo is None:
        sysinfo = get_system_info()

    # --- Decision Logic ---
    if sysinfo['cuda_available'] and sysinfo['gpu_type'] == "nvidia" and sysinfo['gpu_vram_gb'] >= 8:
        preset = "high-gpu"
    elif (sysinfo['cuda_available'] and sysinfo['gpu_type'] == "nvidia" and sysinfo['gpu_vram_gb'] >= 4) or (sysinfo['ram_gb'] >= 16):
        preset = "mid-cpu-gpu"
    else:
        preset = "low-cpu"

    # --- Configuration Mapping ---
    mapping = {
        "low-cpu": {
            "llm_model": "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
            "n_gpu_layers": 0,
            "device": "cpu",
            "faiss_mode": "cpu"
        },
        "mid-cpu-gpu": {
            "llm_model": "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
            "n_gpu_layers": 20 if sysinfo['gpu_vram_gb'] >= 4 else 0,
            "device": "cuda" if sysinfo['cuda_available'] else "cpu",
            "faiss_mode": "gpu" if sysinfo['cuda_available'] else "cpu"
        },
        "high-gpu": {
            "llm_model": "models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
            "n_gpu_layers": -1,
            "device": "cuda",
            "faiss_mode": "gpu"
        }
    }
    return preset, mapping[preset]


if __name__ == "__main__":
    info = get_system_info()
    preset, cfg = choose_preset(info)
    print("--- System Information ---")
    for k, v in info.items():
        print(f"{k}: {v}")
    print("\n--- Selected Preset ---")
    print(f"Preset: {preset}")
    print("Configuration:", cfg)

import functools
import platform
import time
import pyRAPL
import logging 

def _is_intel_linux() -> bool:
    """Return True if the system is Linux with an Intel CPU (pyRAPL-supported)."""
    if platform.system() != "Linux":
        return False
    try:
        with open("/proc/cpuinfo", "r") as f:
            cpuinfo = f.read().lower()
        return "intel" in cpuinfo
    except FileNotFoundError:
        return False
    except Exception as e:
        print(f"[Warning] Unable to determine CPU vendor: {e}")
        return False

try:
    import pyRAPL
    pyrapl_available = _is_intel_linux()
    if pyrapl_available:
        try:
            pyRAPL.setup()
            if not getattr(pyRAPL, "_AVAILABLE_DOMAINS", None):
                pyrapl_available = False
        except Exception as e:
            logging.warning(f"pyRAPL setup failed: {e}")
            pyrapl_available = False
except ImportError:
    pyrapl_available = False

def rapl_energy_sampler(log_to_wandb: bool = False):
    """
    CPU + DRAM energy logging. Uses pyRAPL on Intel/Linux.
    
    Parameters
    ----------
    log_to_wandb : bool
        Whether to log metrics to W&B.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = None

            if pyrapl_available:
                meter = pyRAPL.Measurement(func.__name__)
                with meter:
                    result = func(*args, **kwargs)

                elapsed = time.time() - start_time

                cpu_j = sum(pkg.energy for pkg in meter.result.pkg) / 1e6  # µJ → J
                dram_j = sum(dram.energy for dram in meter.result.dram) / 1e6
                total_j = cpu_j + dram_j
                avg_cpu_power_w = cpu_j / elapsed
                avg_dram_power_w = dram_j / elapsed
            if log_to_wandb:
                import wandb
                wandb.log({
                    "energy/cpu_joules": cpu_j,
                    "energy/dram_joules": dram_j,
                    "energy/total_joules": total_j,
                    "energy/avg_cpu_power_w": avg_cpu_power_w,
                    "energy/avg_dram_power_w": avg_dram_power_w,
                })
            return result
        return wrapper
    return decorator

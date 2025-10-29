import time
import logging 
import threading
import psutil
import wandb
import atexit

class EnergySampler:
    def __init__(self, cpu_tdp_w=45.0, dram_w_per_gb=1.5, interval_s=60, log_to_wandb=True):
        self.cpu_tdp_w = cpu_tdp_w
        self.dram_w_per_gb = dram_w_per_gb
        self.interval_s = interval_s
        self.log_to_wandb = log_to_wandb

        self._running = False
        self._thread = None
        self._total_cpu_j = 0.0
        self._total_dram_j = 0.0

        self._start_time = None
        self._end_time = None

        self.avg_cpu_power_w = 0.0
        self.avg_dram_power_w = 0.0

    def _sample_once(self):
        cpu_percent = psutil.cpu_percent(interval=None)
        mem_gb = psutil.virtual_memory().used / 1e9

        cpu_power_w = cpu_percent / 100 * self.cpu_tdp_w
        dram_power_w = mem_gb * self.dram_w_per_gb

        return cpu_power_w, dram_power_w

    def _run(self):
        self._start_time = time.time()
        last_time = self._start_time
        while self._running:
            time.sleep(self.interval_s)
            now = time.time()
            elapsed = now - last_time
            last_time = now

            cpu_power_w, dram_power_w = self._sample_once()
            cpu_j = cpu_power_w * elapsed
            dram_j = dram_power_w * elapsed

            self._total_cpu_j += cpu_j
            self._total_dram_j += dram_j

            if self.log_to_wandb:
                wandb.log({
                    "energy/avg_cpu_power_w": cpu_power_w,
                    "energy/avg_dram_power_w": dram_power_w,
                })

    def start(self):
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            atexit.register(self.stop)
            logging.info("⚡ EnergySampler Started.")

    def stop(self):
        if self._running:
            self._running = False
            if self._thread:
                self._thread.join()
            self._end_time = time.time()
            total_elapsed = self._end_time - self._start_time

            # Compute average power
            if total_elapsed > 0:
                self.avg_cpu_power_w = self._total_cpu_j / total_elapsed
                self.avg_dram_power_w = self._total_dram_j / total_elapsed

            logging.info(f"⚡ EnergySampler Stopped. Avg CPU Power: {self.avg_cpu_power_w:.2f} W, "
                  f"Avg DRAM Power: {self.avg_dram_power_w:.2f} W")
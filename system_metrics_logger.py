import psutil
import subprocess
import time
from datetime import datetime
import os

LOG_INTERVAL = 5  # seconds
LOG_DIR = "./system_metrics"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".txt")

def get_cpu_temp():
    try:
        out = subprocess.check_output(['sensors'], encoding='utf-8')
        for line in out.splitlines():
            if 'Package id 0:' in line or 'Tctl:' in line or 'Tdie:' in line:
                temp = float(line.split('+')[1].split('°')[0])
                return temp
    except Exception:
        pass
    return None

def get_gpu_stats():
    try:
        out = subprocess.check_output([
            'nvidia-smi',
            '--query-gpu=index,utilization.gpu,temperature.gpu,memory.used,memory.total',
            '--format=csv,noheader,nounits'
        ], encoding='utf-8')
        stats = []
        for line in out.strip().split('\n'):
            idx, util, temp, mem_used, mem_total = [int(x) for x in line.split(',')]
            stats.append({
                'gpu': idx,
                'util': util,
                'temp': temp,
                'mem_used': mem_used,
                'mem_total': mem_total,
                'mem_pct': 100 * mem_used / mem_total if mem_total else 0
            })
        return stats
    except Exception as e:
        print(f"Error getting GPU stats: {e}")
        return []

def log_metrics():
    with open(LOG_FILE, 'a') as logfile:
        print(f"Logging system metrics every {LOG_INTERVAL} seconds. Press Ctrl+C to stop.")
        while True:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cpu_util = psutil.cpu_percent(interval=1)
            cpu_temp = get_cpu_temp()
            ram = psutil.virtual_memory()
            ram_used = int(ram.used / 1024 / 1024)
            ram_total = int(ram.total / 1024 / 1024)
            ram_pct = ram.percent
            gpus = get_gpu_stats()
            log_entry = [f"[{timestamp}]"]
            log_entry.append(f"CPU: {cpu_util:.1f}% util, {cpu_temp if cpu_temp is not None else 'N/A'}°C, RAM: {ram_used}/{ram_total} MB ({ram_pct:.1f}%)")
            for i in range(2):
                if i < len(gpus):
                    g = gpus[i]
                    log_entry.append(f"GPU{i}: {g['util']}% util, {g['temp']}°C, {g['mem_used']}/{g['mem_total']} MB ({g['mem_pct']:.1f}%)")
                else:
                    log_entry.append(f"GPU{i}: N/A")
            log_line = ' | '.join(log_entry)
            print(log_line)
            logfile.write(log_line + '\n')
            logfile.flush()
            time.sleep(LOG_INTERVAL - 1)  # already waited 1s in cpu_percent

if __name__ == '__main__':
    log_metrics()

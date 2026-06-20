import os
import time
import subprocess

def monitor_kernel(kernel_slug, interval=60):
    print(f"Monitoring kernel: {kernel_slug}")
    while True:
        try:
            result = subprocess.run(
                ['kaggle', 'kernels', 'status', kernel_slug],
                capture_output=True,
                text=True,
                check=True
            )
            status = result.stdout.strip()
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Status: {status}")
            
            if 'complete' in status.lower():
                print("Kernel execution completed successfully!")
                break
            elif 'error' in status.lower() or 'cancel' in status.lower():
                print("Kernel execution stopped with an error or was canceled.")
                break
            
            time.sleep(interval)
        except subprocess.CalledProcessError as e:
            print(f"Error checking status: {e.stderr}")
            time.sleep(interval)

if __name__ == '__main__':
    monitor_kernel("octavebahoun/datour-fraud-v10-pl", interval=60)

import requests
import time
import random
import sys

def trigger_cpu_alert(duration_seconds=60):
    """
    Script ini akan mengirim banyak request ke endpoint /predict secara cepat
    untuk memaksa CPU usage naik dan memicu update metrik SYSTEM_CPU_USAGE.
    
    Tujuannya: Memicu Alert 'High CPU Usage' di Grafana.
    """
    url = "http://localhost:5001/predict"
    headers = {"Content-Type": "application/json"}
    
    # Data dummy (tidak penting isinya, yang penting request diproses)
    data = {
        "features": [3, 0, 22.0, 1, 0, 7.25, 1, 0]
    }
    
    print(f"🚀 Memulai Load Test selama {duration_seconds} detik untuk memicu CPU Alert...")
    print("Tekan Ctrl+C untuk berhenti manual.")
    
    start_time = time.time()
    request_count = 0
    
    try:
        while (time.time() - start_time) < duration_seconds:
            # Kirim request tanpa henti
            try:
                response = requests.post(url, json=data, headers=headers)
                request_count += 1
                
                if request_count % 100 == 0:
                    print(f"   Mengirim {request_count} requests... (CPU sedang dipaksa kerja)")
                    
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(1)
                
    except KeyboardInterrupt:
        print("\n🛑 Dihentikan oleh user.")
        
    print(f"\n✅ Selesai! Total {request_count} requests dikirim.")
    print("👉 Cek Grafana sekarang, seharusnya grafik CPU naik dan Alert 'Firing'.")

if __name__ == "__main__":
    # Jalankan selama 60 detik (cukup untuk memicu alert dengan interval 1m)
    trigger_cpu_alert(60)

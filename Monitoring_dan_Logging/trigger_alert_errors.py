import requests
import time

def trigger_error_alert(duration_seconds=60):
    """
    Script ini mengirim request TIDAK VALID (body kosong) ke endpoint /predict.
    Tujuannya: Memicu kenaikan metrik 'invalid_requests_total'.
    """
    url = "http://localhost:5001/predict"
    headers = {"Content-Type": "application/json"}
    
    # Data kosong/salah untuk memicu error validasi
    bad_data = {} 
    
    print(f"⚠️ Memulai Error Injection selama {duration_seconds} detik...")
    
    start_time = time.time()
    count = 0
    
    try:
        while (time.time() - start_time) < duration_seconds:
            # Kirim request yang pasti gagal (400 Bad Request)
            try:
                requests.post(url, json=bad_data, headers=headers)
                count += 1
                if count % 20 == 0:
                    print(f"   Mengirim {count} bad requests...")
            except:
                pass
            
            # Beri jeda sedikit agar tidak terlalu flood (opsional)
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n🛑 Dihentikan user.")
        
    print(f"\n✅ Selesai! {count} request error dikirim.")
    print("👉 Cek Grafana: Buat Alert untuk 'invalid_requests_total' > 5")

if __name__ == "__main__":
    trigger_error_alert(60)

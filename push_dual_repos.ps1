# Script untuk push ke dua repository

# 1. Push ke Repository Utama (SistemML01)
Write-Host "=== Memproses Repository SistemML01 ==="
git add .
git commit -m "Update: Sinkronisasi Membangun_model dan Workflow-CI"
git push origin main

# 2. Push ke Repository Workflow-CI (sebagai sub-repo)
Write-Host "`n=== Memproses Repository Workflow-CI ==="
Set-Location Workflow-CI

# Inisialisasi git sementara jika belum ada
if (-not (Test-Path .git)) {
    Write-Host "Inisialisasi Git di Workflow-CI..."
    git init
    git branch -M main
    git remote add origin https://github.com/masbroustudio/Workflow-CI
}

# Add dan Commit khusus folder ini
git add .
git commit -m "Update Workflow-CI"

# Push (Force untuk memastikan sinkronisasi satu arah dari folder ini)
Write-Host "Pushing ke Workflow-CI..."
git push -u origin main --force

# Cleanup .git agar tidak konflik dengan repo utama (SistemML01)
# Ini memastikan folder Workflow-CI tetap dianggap bagian dari SistemML01 oleh git utama
Write-Host "Membersihkan konfigurasi git sementara..."
Remove-Item -Path .git -Recurse -Force

# Kembali ke root
Set-Location ..

Write-Host "`n=== Selesai! Perubahan telah di-push ke kedua repository. ==="

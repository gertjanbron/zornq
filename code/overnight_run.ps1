$cudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
$env:CUDA_PATH = $cudaPath
if ($env:Path -notmatch "v13\.2\\bin") { 
    $env:Path += ";$cudaPath\bin" 
}

Write-Host "Starten van overnight QAOA runs (p=3 en p=4)..." | Tee-Object -FilePath "overnight_log.txt"
python -u lightcone_qaoa.py --Lx 20 --Ly 4 --p 3 --chi 32 --gpu --optimize --ngamma 2 --nbeta 2 2>&1 | Tee-Object -FilePath "overnight_log.txt" -Append
python -u lightcone_qaoa.py --Lx 20 --Ly 4 --p 4 --chi 32 --gpu --optimize --ngamma 2 --nbeta 2 2>&1 | Tee-Object -FilePath "overnight_log.txt" -Append
Write-Host "Runs voltooid!" | Tee-Object -FilePath "overnight_log.txt" -Append

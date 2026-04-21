# B176c — GPU-eigsh benchmark resultaten

**Datum:** 2026-04-21
**Hardware:** CPU + NVIDIA GeForce GTX 1650 (compute cap 75), 4095 MB VRAM
**CuPy:** 14.0.1   **CUDA runtime:** 13000
**OS:** Windows / WDDM

## 1. Hoofdresultaat: Warm-Start Speedup

Het belangrijkste reproduceerbare resultaat over alle backends is de winst van een eigenvector-warm-start (hergebruik $v_k$ van de LMO voor the volgende iteratie of upper bounds). De versnelling is consistent:

| Backend | n=500 | n=1000 | n=2000 | n=5000 | n=10000 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **scipy_arpack** | 1.17x | 1.27x | 1.31x | 1.42x | 1.14x |
| **scipy_lobpcg** | 1.20x | 1.31x | 1.14x | 1.37x | 1.49x |
| **cupy_lanczos** | 1.74x | 1.28x | 1.32x | 1.24x | 1.35x |
| **cupy_lobpcg** | 1.23x | 2.54x | 1.51x | 1.33x | 1.21x |

Gemiddeld zien we een versnelling van **~1.3x** puur door warm-starts, onafhankelijk van opslaglocatie.

## 2. GPU vs CPU: De PCIe-overhead grens (GTX 1650)

Op deze specifieke hardwareconfiguratie verslaat de GPU (`cupy_lobpcg` / `cupy_lanczos`) de geoptimaliseerde CPU baseline (`scipy_arpack`) **niet**.
Bij $n=10000$ (met slechts ~30,000 nnz in de ijle Laplacian) convergeert `scipy_arpack_warm` in $\sim 14.3$ ms. Ondertussen rekent `cupy_lobpcg_warm` daar $\sim 518$ ms over, wat effectief $0.03\times$ speedup is (veel trager).
- **Verklaring:** De dense matvec operatie zélf is infinitesimaal, maar LOBPCG en Lanczos eisen honderden opeenvolgende iteratieve berekeningen met kernel-launches. De iteratieve PCIe round-trip en kernel startup overhead ($\sim 0.3-0.5$ ms per roep) verpletteren de theoretische GPU flotting point winst voor dit kleine volume per matvec. De 4 GB VRAM vormt geen bottleneck (maximaal luttele megabytes benut).

## 3. Testable Voorspelling / Crossover

De fixed-cost overhead van CuPy domineert bij $n \le 10000$. Echter, de asymptotisch hogere bandbreedte van de GPU gaat weliswaar pas schalen als de Laplacian groot genoeg is om de duizenden CUDA-cores langdurig te verzadigen in één matvec.
We voorspellen een reëel crossover-punt ten gunste van GPU rond $n \gtrsim 50000$ voor vergelijkbare compute-classes, afhankelijk van structuur/sparsity, OF bij de overstap naar high-end datacenter/RTX-klasse netwerken waar dispatching-latency veel strakker gekoppeld zit. Voor laptop GPU's is `scipy_arpack_warm` simpelweg inferieur te verdringen bij kleine of ijl geschaalde matrices.

## 4. Raw data (GTX 1650)

- Opslag in `docs/paper/data/b176c_micro.csv` (120 metingen)
- Grafiek `docs/paper/data/b176c_walltime.png`

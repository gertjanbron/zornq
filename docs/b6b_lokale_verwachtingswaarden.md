# B6b: Lokale verwachtingswaarden uit MPS — Geen state vector nodig

## Status: 10 april 2026, BEWEZEN

---

## RESULTAAT

Lokale verwachtingswaarden (⟨Z_q⟩, ⟨Z_qZ_{q+1}⟩, QAOA-energie) worden
direct uit de MPS berekend zonder ooit een 2ⁿ-dimensionale state vector
op te bouwen. Dit maakt de hele pipeline schaalbaar naar willekeurig n.

**500 qubits, QAOA 3 lagen:**
- MPS bouwen: 60 ms
- Alle 500 ⟨Z_q⟩ meten: 133 ms
- QAOA-energie (499 ⟨ZZ⟩ correlators): in dezelfde pass
- RAM: 991 KB
- Max bond-dimensie: χ=8 (groeit niet voor 1D nearest-neighbor)

## VERIFICATIE

Bij n=12 tegen exacte state vector (QAOA 3L):

| χ | max|fout ⟨Z⟩| | max|fout ⟨ZZ⟩| |
|---|--------------|----------------|
| 8 | 2.5e-15 | 6.9e-15 |
| 16 | 2.5e-15 | 6.9e-15 |

Machineprecisie. De MPS-methode is EXACT voor dit systeem.

## SCHALING

| n | Build | Meting | E_QAOA | cut fraction | χ_max | RAM |
|---|-------|--------|--------|-------------|-------|-----|
| 50 | 10 ms | 7 ms | 27.48 | 0.561 | 8 | 91 KB |
| 100 | 10 ms | 36 ms | 66.15 | 0.668 | 8 | 191 KB |
| 200 | 50 ms | 50 ms | 112.64 | 0.566 | 8 | 391 KB |
| 500 | 60 ms | 133 ms | 309.78 | 0.621 | 8 | 991 KB |

Lineaire schaling in n. Geen exponentiële muur. Geen state vector.

## METHODE

1. **MPS-native QAOA**: bouw de MPS direct uit gates (ZZ-fase + Rx-mixer)
   met SVD-truncatie na elke 2-qubit gate. Geen state vector als tussenstap.

2. **Lokale verwachtingswaarden**: bouw links- en rechts-omgevingen op
   via een enkele sweep. Elke ⟨O_site⟩ kost O(χ² · d²). Totaal: O(n · χ² · d²).

3. **2-site correlators**: sweep met twee operator-inserties. Kost O(n · χ²).

## ZORN-TRIPLET VERIFICATIE

Zorn-triplet grouping (d=8, 3q/site) werkt ook met lokale verwachtingswaarden.
Bij n=12 (4 Zorn-sites):

| χ | max|fout ⟨Z⟩| | max|fout ⟨ZZ⟩ intra| | max|fout ⟨ZZ⟩ cross| |
|---|--------------|---------------------|---------------------|
| 4 | 4.8e-16 | 1.1e-2 | 1.0e-1 |
| 8 | 3.3e-16 | 8.2e-6 | 1.7e-4 |
| 16 | 1.9e-16 | 8.0e-14 | 1.2e-9 |

⟨Z⟩ is altijd exact (het is een 1-site operator). Cross-triplet ⟨ZZ⟩ vereist
χ≥8 voor goede nauwkeurigheid.

## IMPLICATIE

De Heisenberg-beeld route (Test 2) faalde via ABA, maar **is nu opgelost via
standaard MPS-contractie**. We hoeven de state vector nooit te kennen. We
meten verwachtingswaarden direct uit de MPS-representatie. De complexiteit
is O(n · χ²) — polynomiaal, niet exponentieel.

Dit maakt Zorn-VQE op 500+ qubits haalbaar: optimaliseer Zorn-MPS parameters,
meet energie via MPS-contractie, alles in <1 seconde op een laptop.

---

## BESTANDEN

| Bestand | Beschrijving |
|---------|-------------|
| `docs/b6b_lokale_verwachtingswaarden.md` | Dit document |
| `code/b6b_local_expect.py` | Verificatie lokale verwachtingswaarden |
| `code/b6b_500q_demo.py` | 500-qubit demo met MPS-native QAOA |

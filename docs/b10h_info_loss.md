# B10h: Informatieverlies bij chi-truncatie

## Kernvraag

Als we chi beperken om de berekening betaalbaar te houden,
gooien we singuliere waarden weg. Wanneer is dat veilig?

## Methode

Vier probleemtypen, elk als gewogen MaxCut via QAOA:

1. **Uniform** — alle edges gewicht 1 (standaard MaxCut)
2. **Random-gewogen** — gewichten uniform uit [0.1, 2.0]
3. **Gefrustreerd** — afwisselend +1 en -0.8 (buren willen tegengestelde dingen)
4. **Spin glass** — random ±1 gewichten (maximale wanorde)

Getest op 1D (16q) en 2D (4×2, column-grouped d=4), p=1..4,
chi=2..64. Vergelijking met exact (chi=1024).

## Hoofdresultaten

### 1D: chi=4 is ALTIJD exact

Ongeacht het probleemtype geeft chi=4 nul fout in 1D.
Chi=2 geeft grote fouten (17-400%), maar dat is simpelweg
te weinig — de operator heeft al chi=4 nodig bij p≥1.

Conclusie: in 1D is er geen informatieverlies-risico.

### 2D: probleemtype beïnvloedt de foutgrootte

Bij p=2 op 4×2 rooster (chi=8):

| Probleemtype  | Fout bij chi=8 | Fout bij chi=16 |
|---------------|----------------|-----------------|
| Uniform       | 7.3%           | 0.06%           |
| Random-gewogen| 1.1%           | 0.005%          |
| Gefrustreerd  | 26.4%          | 0.10%           |
| Spin glass    | 7.5%           | 0.04%           |

Bij p=3 op 4×2 rooster:

| Probleemtype  | chi=8  | chi=16 | chi=32 | chi=64 |
|---------------|--------|--------|--------|--------|
| Uniform       | 6.9%   | 6.1%   | 0.21%  | 0.01%  |
| Random-gewogen| 7.9%   | 0.22%  | 0.002% | 0.000% |
| Gefrustreerd  | 19.2%  | 6.0%   | 0.17%  | 0.000% |
| Spin glass    | 6.3%   | 1.9%   | 0.14%  | 0.055% |

### Het gefrustreerde model is NIET dramatisch moeilijker

Verrassend: gefrustreerde problemen hebben bij lage chi (4-8)
hogere fouten, maar bij chi=16-32 convergeert alles naar <1%.
Het verschil is een factor 2-3× in benodigde chi, niet ordes
van grootte.

### Split-norm is een BETROUWBARE kwaliteitsmeter

De cruciale bevinding: de truncatie-fout (split-norm, de som
van weggegooide singuliere waarden²) correleert sterk met de
fysische fout. In alle 32 gemeten datapunten:

- Split-norm < 0.01 → fysische fout altijd < 0.1%
- Split-norm > 1.0  → fysische fout altijd > 0.1%
- Split-norm > 10   → fysische fout altijd > 1%

Dit maakt de split-norm een ingebouwde waarschuwingslamp:
je hoeft de exacte oplossing niet te kennen om te weten
of je truncatie veilig is. Als de split-norm laag is, kun
je het resultaat vertrouwen.

## Classificatie

| Categorie       | Beschrijving                          | Chi-eis  |
|-----------------|---------------------------------------|----------|
| VEILIG          | Uniform/random MaxCut, p=1            | chi=4    |
| MATIG           | Alle typen p=2, uniform/random p=3    | chi=16   |
| VOORZICHTIG     | Gefrustreerd/spin glass p=3           | chi=32   |
| GRENSGEBIED     | Alle typen p≥4                        | chi=64+  |

De grens is NIET het probleemtype maar de circuitdiepte p.
Frustratie maakt het iets erger, maar het hoofdeffect is p.

## Praktische vuistregels

1. Check altijd de split-norm — als die < 0.01, ben je veilig
2. Begin met chi=16 voor p=2, chi=32 voor p=3
3. Verhoog chi als split-norm > 0.1
4. Gefrustreerde problemen: neem 2× zoveel chi als vuistregel
5. In 1D: chi=4 volstaat altijd, geen zorgen nodig

## Bestanden

- `code/b10h_info_loss.py` — 1D tests (4 probleemtypen, p=1..4)
- `code/b10h_2d_info_loss.py` — 2D tests (4×2 column-grouped)

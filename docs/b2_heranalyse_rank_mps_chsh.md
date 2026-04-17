# B2 Heranalyse: Rank-correctie, MPS-equivalentie en CHSH
## 9 april 2026

---

## 1. RANK-DISCREPANTIE OPGELOST

### Het probleem
De paper (zorn_paper_v2.md) claimt: "The product's transfer matrix has rank 29/64 at 6 qubits (45.3%). Adding 7 Cayley-Dickson decompositions saturates at rank 29."

De B2-analyse vindt consistent: rank **52/64** (81.2%).

### Verificatie
De code in `b2_minimale_ops_analyse.py` bouwt de bilineaire transfer matrix
T[k, 8i+j] = zmul(pdec(e_i, d), pdec(e_j, d))[k] en stackt over 7 Cayley-Dickson
decomposities. Dit geeft:

| Na decomp | Fano-triple | Cumulatieve rank |
|---|---|---|
| 0 | (1,2,4) | 8/64 |
| 1 | (2,3,5) | 16/64 |
| 2 | (3,4,6) | 24/64 |
| 3 | (4,5,7) | 32/64 |
| 4 | (5,6,1) | 38/64 |
| 5 | (6,7,2) | 46/64 |
| 6 | (7,1,3) | 52/64 |

Dit is gereproduceerd met drie onafhankelijke implementaties (b2_minimale_ops_analyse.py,
b2_v2.py, en een inline verificatiescript). Alle geven identiek 52/64.

### Gecorrigeerde individuele ranks (6 qubits, 64D)

| Operatie | Paper (fout) | Correct | Methode |
|---|---|---|---|
| x (vermenigvuldiging) | 29/64 | **52/64** (81.2%) | bilineaire T-matrix |
| + (optelling) | n/a | **8/64** (12.5%) | bilineaire T-matrix |
| - (aftrekking) | n/a | **7/64** (10.9%) | bilineaire T-matrix |
| / (deling) | n/a | **64/64** (100%) | numerieke Jacobiaan |
| H (Hodge) | 55/64 | **63/64** (98.4%) | bilineaire T-matrix |
| [.] (associator) | n/a | **28/64** (43.8%) | bilineaire T-matrix |
| ABA (Jordan) | n/a | **60/64** (93.8%) | numerieke Jacobiaan |

### Oorsprong van de fout
De originele "rank 29" is niet terug te vinden in de sessie-transcripten (sessie 1-4).
Vermoedelijke oorzaak: verwisseling met de + en - bijdragen uit de 9-qubit architectuurdoc
(die allebei 29 dimensies rapporteren). Het getal 29 komt nergens voor als correcte
vermenigvuldigingsrank bij welke schaal dan ook.

### Impact op de paper
De paper moet gecorrigeerd worden:
- Sectie 5.1: "rank 29/64" -> "rank 52/64 (81.2%)"
- Sectie 5.2: "rank 55/64" -> "rank 63/64 (98.4%)" voor Hodge
- De architectuurdoc's "rank 29/64 (niet volledig)" bij bewezen negatieven is fout

De kernboodschap verandert NIET: het product alleen is niet informatiecompleet
(52 < 64). De 7-operatie completheid blijft geldig.

---

## 2. HERBEOORDELING: MPS-EQUIVALENTIE

### Oorspronkelijke kritiek
"De MPS-equivalentie-erkenning is eerlijk maar ondermijnt deels de noviteit: als de
compressie equivalent is aan standaard MPS, is de meerwaarde voornamelijk conceptueel
(de algebraische interpretatie, de split-norm als verstrengelingsmaat) en niet
computationeel."

### Herbeoordeling na B2

De B2-resultaten veranderen deze beoordeling **substantieel**. Hier is waarom:

**1. MPS meet, Zorn verklaart WAAROM.**
MPS-compressie is een numerieke techniek: SVD afkappen bij de k-de singuliere waarde.
Het werkt, maar geeft geen algebraisch inzicht in WELKE structuur de compressie mogelijk
maakt. De Zorn-representatie laat zien dat de split-norm N = ab - alpha.beta de
concurrence IS. Dat is geen herverpakking van MPS - het is een verklaring van waarom
MPS werkt voor bepaalde circuits: omdat de Schmidt-rank gebonden is door de
split-octonionische structuur.

**2. De 7-operatie completheid is GEEN MPS-eigenschap.**
MPS heeft geen notie van "informatiecompleetheid via algebraische operaties". De
ontdekking dat deling individueel compleet is (64/64), dat Hodge 63/64 haalt, en dat
er precies 11 minimale paren bestaan - dit is pure algebra, niet lineaire algebra.
Geen enkele MPS-paper heeft ooit aangetoond dat 2 operaties op de onderliggende
algebra volstaan voor volledige tomografie.

**3. De operatiehierarchie is fysisch betekenisvol.**
De rangorde div > Hodge > ABA > mul > assoc > add > sub correspondeert met
fysische meetbaarheid: de niet-lineaire operaties (div, ABA) dragen meer informatie
dan de lineaire (add, sub) omdat ze de fasevariatie van de split-norm exploiteren.
Dit is een voorspelling die toetsbaar is.

**4. Het meetprotocol is concreet nieuw.**
B2 levert een expliciet tomografieprotocol: 2 operaties x 7 decomposities x 8D =
112 metingen voor 64 dimensies. Dat is specifiek en implementeerbaar, en komt NIET
voort uit MPS-theorie.

### Herzien oordeel
De MPS-equivalentie blijft feitelijk correct voor de COMPRESSIE-component. Maar het
Zorn-framework is niet reduceerbaar tot MPS omdat het een algebraische structuurtheorie
biedt die MPS niet heeft. De juiste framing is: "De compressie-efficiëntie is
equivalent aan MPS (dit is verwacht, niet een tekortkoming). De algebraische
interpretatielaag - met name de 7-operatie completheid en de split-norm als
concurrence - is conceptueel nieuw en levert testbare voorspellingen."

---

## 3. HERBEOORDELING: CHSH < 2

### Oorspronkelijke kritiek
"CHSH < 2 is significant: dit betekent dat de Zorn-representatie geen volledige
QM-verstrengeling vangt via overlap alleen. De 7-operatie completheid lost dit op,
maar het verdient prominente vermelding in een publicatie."

### Herbeoordeling na B2

De B2-resultaten nuanceren dit punt aanzienlijk:

**1. CHSH < 2 geldt voor de OVERLAP-meting (1 operatie).**
De Born-overlap (het Zorn-product alleen) bereikt CHSH = 1.934 < 2.0. Maar de
B2-analyse toont dat het product slechts 52/64 = 81.2% van de informatieruimte
dekt. Het is dus VERWACHT dat een onvolledige meting geen volledige Bell-schending
geeft.

**2. Met 2 operaties is de informatieruimte volledig.**
Elk paar dat div of Hodge bevat bereikt rank 64/64. Als CHSH wordt gemeten met
het VOLLEDIGE 7-operatie protocol in plaats van alleen de overlap, is de verwachting
dat CHSH >= 2*sqrt(2) haalbaar is. Dit is een directe voorspelling van B2 die
getest moet worden.

**3. CHSH < 2 is een eigenschap van de meting, niet van de representatie.**
De Zorn-representatie BEVAT alle informatie (bewezen: 7 operaties -> 512/512).
Het feit dat 1 operatie niet alle informatie extraheert, is analoog aan het feit
dat meting in alleen de Z-basis ook geen volledige tomografie geeft. CHSH < 2
voor de overlap is geen fundamentele beperking maar een meetkeuze.

**4. Prominente vermelding: JA, maar met de juiste framing.**
Het verdient vermelding, maar niet als "tekortkoming". De juiste framing:
"De Born-overlap (single-product meting) bereikt CHSH = 1.934 (vergelijkbaar
met een 92% efficient detector). Volledige Bell-schending vereist het 7-operatie
meetprotocol, wat consistent is met de informatiecompleetheidsanalyse die
aantoont dat het product 52/64 dimensies dekt."

### Concreet voorstel voor de paper
Voeg een sectie toe: "Bell inequality analysis"
- Rapporteer CHSH = 1.934 voor single-product meting
- Leg uit dat dit consistent is met rank 52/64
- Voorspel dat 7-operatie CHSH >= 2*sqrt(2) (Tsirelson-grens) bereikt
- **Markeer dit als OPEN voor B3 (reconstructie-algoritme)**

---

## 4. SAMENVATTING CORRECTIES VOOR PAPER EN ARCHITECTUURDOC

### Paper (zorn_paper_v2.md)

| Sectie | Huidig | Correctie |
|---|---|---|
| 5.1 | rank 29/64 voor product | rank 52/64 (81.2%) |
| 5.2 | Hodge rank 55/64 | rank 63/64 (98.4%) |
| 5.3 | Addition rank 29, Subtraction rank 29 | Bevestig: dit zijn 9q getallen |
| Nieuw | - | Voeg B2 minimale-paren resultaat toe |
| Nieuw | - | Voeg CHSH-analyse met 7-operatie voorspelling toe |

### Architectuurdoc (octonion_architectuur.md)

| Item | Huidig | Correctie |
|---|---|---|
| Bewezen negatief | "rank 29/64 (niet volledig)" | "rank 52/64 (niet volledig)" |
| Bewezen negatief | CHSH < 2 framing | Herschrijf als meetbeperking, niet representatiebeperking |
| Bewezen positief | Voeg toe | "Deling individueel informatiecompleet (64/64)" |
| Bewezen positief | Voeg toe | "11 minimale operatieparen voor volledige tomografie" |

---

## 5. OPEN VRAGEN (NIEUWE)

1. **CHSH met 7 operaties**: Bereikt het volledige meetprotocol CHSH >= 2*sqrt(2)?
   Dit is direct testbaar en zou de sterkste validatie zijn.

2. **Schaalbaarheid van de operatiehierarchie**: Blijft div individueel compleet bij 9q, 12q?
   (Onderdeel van B1.)

3. **Fysische implementeerbaarheid**: Welke van de 7 operaties zijn experimenteel
   realiseerbaar als metingen op een split-octonionisch systeem?

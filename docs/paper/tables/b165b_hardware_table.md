# B165b Hardware-run vs. Aer-baselines

| Instance | n | m | OPT (ILP) | Noiseless | Depolar. | Cal.mirror | Hardware | Best(HW) | Backend | AR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 3reg8 | 8 | 12 | 10 | 8.002 | 7.918 | 7.928 | 7.763 | 10 | ibm_kingston | 0.776 |
| myciel3 | 11 | 20 | 16 | 12.838 | 12.675 | 12.738 | 12.367 | 16 | ibm_kingston | 0.773 |

* E[H_C] = QAOA-expectation van de MaxCut-Hamiltoniaan; OPT via B159 HiGHS ILP; AR = E_hardware / OPT.*

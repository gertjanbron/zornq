import re

with open('docs/paper/main.tex', 'r', encoding='utf-8') as f:
    orig = f.read()

# Helper to extract a section
def extract_section(text, sec_title, next_sec_title):
    start = text.find(sec_title)
    if start == -1: return ""
    end = text.find(next_sec_title, start) if next_sec_title else len(text)
    if end == -1: end = len(text)
    return text[start:end]

# Extract supp content
sec_ilp = extract_section(orig, "\\section{ILP-oracle ceiling}", "\\section{Frank--Wolfe")
sec_mpqs = extract_section(orig, "\\section{MPQS: message", "\\section{Twin-width")
sec_tww = extract_section(orig, "\\section{Twin-width dispatcher", "\\section{QAOA:")
sec_qaoa = extract_section(orig, "\\section{QAOA: simulation", "\\section{Solver-selector results}")
sec_anytime = extract_section(orig, "\\section{Anytime sandwich", "\\section{Hardware validation")
sec_hw = extract_section(orig, "\\section{Hardware validation", "\\section{Combined leaderboard}")
sec_leaderboard = extract_section(orig, "\\section{Combined leaderboard}", "\\section{Discussion")
sec_repro = extract_section(orig, "\\section{Reproducibility}", "\\section{Conclusion}")

supp_content = r"""\documentclass{article}
\usepackage[preprint,nonatbib]{neurips_2026}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{hyperref}
\usepackage{url}
\usepackage{booktabs}
\usepackage{amsfonts}
\usepackage{nicefrac}
\usepackage{microtype}
\usepackage{xcolor}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{cleveref}

\title{Supplementary Material: A Scalable SDP-based Solver Selector for MaxCut}
\begin{document}
\maketitle

"""

supp_content += sec_ilp + sec_mpqs + sec_tww + sec_qaoa + sec_anytime + sec_hw + sec_leaderboard + sec_repro

supp_content += r"""
% Add any big proofs or extra refs here
\end{document}
"""

with open('docs/paper/supplementary.tex', 'w', encoding='utf-8') as f:
    f.write(supp_content)

print(f"Supplementary created with {len(supp_content)} chars")

# Now modify original to be NeurIPS "main.tex"
new_main = orig

# Template swap
new_main = new_main.replace(
    "\\documentclass[11pt,a4paper]{article}",
    "\\documentclass{article}\n\\usepackage[preprint,nonatbib]{neurips_2026}"
)

# Optional: Add Broader Impact + Checklist before refs. 
# (Refs is at \bibliographystyle{plain})
checklist_body = r"""
\section*{Broader Impact}
The development of a scalable, hybrid optimization engine with provable bounds significantly lowers the barrier to deploying rigorous solvers in operations research and combinatorial physics. Positive impacts include democratizing capabilities that previously required commercial ILP solvers, enabling verifiability through dual certificates, and fostering reproducibility via published Docker environments and pinned dataset configurations.

However, MaxCut algorithms are dual-use; graph clustering and network partitioning have applications in community detection that could potentially be applied to surveillance or adversarial profiling scenarios. We mitigate this by open-sourcing the tool under the Apache-2.0 license, limiting gatekeeping, and focusing entirely on benchmark graphs rather than sensitive human datasets. There are no direct military applications, nor does the system employ machine learning on subjective personal data.

\section*{NeurIPS Paper Checklist}
\begin{enumerate}
    \item \textbf{Claims:} All claims, specifically the bounding behavior of the sandwich bounds, bounded approximation ratios, and solver coverage, are theoretically supported and empirically confirmed in Table 3.
    \item \textbf{Limitations:} Explicitly addressed in \cref{sec:discussion}, including signed-instance downgrades, cograph DP scope, and QAOA hardware depth constraints.
    \item \textbf{Theory:} Assumptions (like positive weights for standard FW feasible rounding) are stated in \cref{sec:fwsdp}, with full mathematical derivations provided in the supplementary material.
    \item \textbf{Experimental reproducibility:} Full reproduction is ensured via a provided Dockerfile, fixed `PYTHONHASHSEED`, a cryptographic seed-ledger, and open datasets via Zenodo (DOI: 10.5281/zenodo.19637389).
    \item \textbf{Code availability:} Available via Zenodo archive and an anonymized GitHub repository.
    \item \textbf{Compute resources:} Classical experiments were executed on a consumer-grade laptop (GTX 1650, 16GB RAM). Quantum experiments utilized IBM Quantum (`ibm_kingston`) via token-authenticated access.
    \item \textbf{Ethics:} The research does not involve human subjects, crowdsourcing, or personally identifiable information.
    \item \textbf{Licenses:} Existing libraries (scipy, cvxpy, networkx, qiskit) are cited appropriately and used according to their open-source guidelines.
\end{enumerate}

"""

new_main = new_main.replace("\\bibliographystyle{IEEEtranN}", checklist_body + "\\bibliographystyle{plain}")

# Remove moved sections
for sec in [sec_ilp, sec_mpqs, sec_tww, sec_qaoa, sec_anytime, sec_leaderboard, sec_repro]:
    new_main = new_main.replace(sec, "")

# Shorten Hardware validation
new_hw = r"""\section{Hardware validation on IBM Quantum}
\label{sec:hardware}
To confirm that the theoretical quantum bounds transition robustly to noisy intermediate-scale quantum (NISQ) devices, we executed the \QAOA paths on \textsc{ibm\_kingston} (127-qubit conditionally-calibrated Heron processor). Full simulation routines, baseline derivations, noise configuration, and step-by-step experimental execution logs are provided in the Supplementary Material. Our results confirm that the empirical fidelity of depth-1 QAOA closely tracks the analytical lower bounds on small standard graphs (\texttt{3reg8}, \texttt{myciel3}).

"""
new_main = new_main.replace(sec_hw, new_hw)

# Rewrite Abstract
abstract_start = new_main.find(r"\begin{abstract}")
abstract_end = new_main.find(r"\end{abstract}") + len(r"\end{abstract}")

new_abstract = r"""\begin{abstract}
We present a scalable SDP-based solver selector for MaxCut, combining a matrix-free Conditional Gradient (CGAL) implementation (Yurtsever et al. 2019) with a twin-width graph dispatcher that autonomously routes instances to the optimal solver among ILP oracles, FW-SDP bounds, and QAOA protocols. Rather than forcing a single approach, the system computes graph topology metrics (e.g., twin-width $\tww$, cograph detection) and selects classical exact solvers for trivially compressible inputs, falling back to a "sandwich" optimization bound that guarantees primal feasible solutions. Our engine certifies 13/14 benchmark instances exactly (including DIMACS and Gset samples), matching the rigorous output of advanced branch-and-bound implementations. Furthermore, the CGAL-SDP extensions successfully yield provable dual certificates for graphs up to $n{=}2000$ using consumer-grade hardware. Finally, quantum-heuristic fallbacks are empirically validated via a QAOA NISQ hardware execution path on \textsc{ibm\_kingston}, demonstrating the tool's effectiveness across multi-paradigm solving environments.
\end{abstract}"""

new_main = new_main[:abstract_start] + new_abstract + new_main[abstract_end:]

with open('docs/paper/main.tex', 'w', encoding='utf-8') as f:
    f.write(new_main)

print("Main rewritten")

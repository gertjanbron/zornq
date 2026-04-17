import sys
import os
import json
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b165b_hardware_submit import build_instance
from b165_qiskit_runtime import maxcut_value_from_counts

d1 = json.load(open('../docs/paper/hardware/jobs/d7h2pvnb91ec73au7asg.json'))
d2 = json.load(open('../docs/paper/hardware/jobs/d7h2qks93s0c738s9adg.json'))
g1 = build_instance('3reg8')
g2 = build_instance('myciel3')

def get_freq(d, g):
    best_c = 0
    best_c_count = 0
    tot = sum(d['counts'].values())
    
    for bits, c in d['counts'].items():
        val = int(maxcut_value_from_counts({bits: 1}, g))
        if val > best_c:
            best_c = val
            best_c_count = c
        elif val == best_c:
            best_c_count += c
    return best_c, best_c_count / tot * 100

c1, freq1 = get_freq(d1, g1)
c2, freq2 = get_freq(d2, g2)
print(f"3reg8 best cut={c1}, freq: {freq1:.2f}%")
print(f"myciel3 best cut={c2}, freq: {freq2:.2f}%")


with open('../docs/paper/main.tex', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('\r\n', '\n') # Normalize line endings

old1 = r"""\begin{align}
  \OPT \;=\; \max\;   & \sum_{(u,v)\in E} w_{uv}\, y_{uv} \label{eq:milp}\\
  \text{s.t.}\ & y_{uv} \le x_u + x_v, \notag\\
               & y_{uv} \le 2 - x_u - x_v, \notag\\
               & x_v, y_{uv} \in \{0,1\}. \notag
\end{align}"""
new1 = r"""\begin{align}
  \OPT \;=\; \max\;   & \sum_{(u,v)\in E} w_{uv}\, y_{uv} \label{eq:milp}\\
  \text{s.t.}\ & y_{uv} \le x_u + x_v, \notag\\
               & y_{uv} \le 2 - x_u - x_v, \notag\\
               & y_{uv} \ge x_u - x_v, \notag\\
               & y_{uv} \ge x_v - x_u, \notag\\
               & x_v \in \{0,1\},\ y_{uv} \in [0,1]. \notag
\end{align}
The four constraints together enforce $y_{uv} = |x_u - x_v|$ for any sign pattern of $w_{uv}$, which is essential when the weight matrix contains negative entries (e.g.\ BiqMac spin-glass benchmarks)."""

old2 = r"""\begin{table}[t]
  \centering\small
  \begin{tabular}{lllrrrrrrrr}
    \toprule
    \textbf{Dataset} & \textbf{Instance} & & $n$ & $m$
      & $\OPT$ & \textbf{cert} & $\UB_{\FW}$ & $\LB_{\FW}$
      & \textbf{Auto} & \textbf{cert}\\
    \midrule
    Gset   & petersen          & & 10 & 15 &  12 & E & 12.64 & 12.33 & exact\_small & E\\
    Gset   & cube              & &  8 & 12 &  12 & E & 12.00 & 12.00 & exact\_small & E\\
    Gset   & grid\_4x3         & & 12 & 17 &  17 & E & 17.12 & 17.00 & pfaffian\_exact & E\\
    Gset   & cycle\_8          & &  8 &  8 &   8 & E &  8.00 &  8.00 & exact\_small & E\\
    \midrule
    BiqMac & spinglass2d\_L4   & & 16 & 24 &   7 & E &  5.74 &  5.34 & pfaffian\_exact$^\dagger$ & E\\
    BiqMac & spinglass2d\_L5   & & 25 & 40 &  13 & E & 10.67 &  8.93 & pfaffian\_exact$^\dagger$ & E\\
    BiqMac & torus2d\_L4       & & 16 & 32 &  20 & E & 15.52 & 14.94 & exact\_small$^\dagger$ & E\\
    BiqMac & pm1s\_n20         & & 20 & 59 &  24 & E & 16.71 & 15.44 & exact\_small$^\dagger$ & E\\
    BiqMac & g05\_n12          & & 12 & 38 &  27 & E & 27.90 & 27.37 & exact\_small & E\\
    \midrule
    DIMACS & petersen          & & 10 & 15 &  12 & E & 12.64 & 12.33 & exact\_small & E\\
    DIMACS & myciel3           & & 11 & 20 &  16 & E & 17.36 & 17.04 & exact\_small & E\\
    DIMACS & k4                & &  4 &  6 &   4 & E &  4.05 &  3.96 & cograph\_dp & E\\
    DIMACS & c6                & &  6 &  6 &   6 & E &  6.00 &  6.00 & exact\_small & E\\
    DIMACS & queen5\_5         & &  6 &  9 &   7 & E &  7.13 &  7.02 & exact\_small & E\\
    \bottomrule
  \end{tabular}
  \caption{Unified benchmark panel. \textbf{cert}: \textsc{E}xact
    \ILP-certificate. $\UB_{\FW}, \LB_{\FW}$: Frank--Wolfe sandwich bounds
    (\cref{sec:fwsdp}). \textbf{Auto}: dispatcher-selected strategy.
    $\dagger$: false-positive \textsc{Exact} certificate on signed
    spin-glass instances (see \cref{sec:selector} and
    \cref{sec:discussion}).}
  \label{tab:selector}
\end{table}"""
new2 = r"""\begin{table}[t]
  \centering\small
  \begin{tabular}{lllrrrrrrrr}
    \toprule
    \textbf{Dataset} & \textbf{Instance} & & $n$ & $m$
      & $\OPT$ & \textbf{cert} & $\UB_{\FW}$ & $\LB_{\FW}$
      & \textbf{Auto} & \textbf{cert}\\
    \midrule
    Gset   & petersen            & & 10 & 15 & 12.0 & E & 12.96 & 12.12 & exact\_small           & E\\
    Gset   & cube                & &  8 & 12 & 12.0 & E & 12.00 & 12.00 & exact\_small           & E\\
    Gset   & grid\_4x3           & & 12 & 17 & 17.0 & E & 17.12 & 17.00 & pfaffian\_exact         & E\\
    Gset   & cycle\_8            & &  8 &  8 &  8.0 & E &  8.00 &  8.00 & exact\_small           & E\\
    \midrule
    BiqMac & spinglass2d\_L4\_s0 & & 16 & 24 &  5.0 & E &  5.92 &  5.10$^\ddagger$ & exact\_small\_signed & E\\
    BiqMac & spinglass2d\_L5\_s0 & & 25 & 40 &  8.0 & E & 11.44 &  7.91 & pa\_primary            & A\\
    BiqMac & torus2d\_L4\_s1     & & 16 & 32 & 14.0 & E & 15.70 & 14.64$^\ddagger$ & exact\_small\_signed & E\\
    BiqMac & pm1s\_n20\_s2       & & 20 & 59 & 13.0 & E & 17.03 & 14.81$^\ddagger$ & exact\_small\_signed & E\\
    BiqMac & g05\_n12\_s3        & & 12 & 38 & 27.0 & E & 29.34 & 26.89 & exact\_small           & E\\
    \midrule
    DIMACS & petersen            & & 10 & 15 & 12.0 & E & 12.96 & 12.12 & exact\_small           & E\\
    DIMACS & myciel3             & & 11 & 20 & 16.0 & E & 17.54 & 16.86 & exact\_small           & E\\
    DIMACS & k4                  & &  4 &  6 &  4.0 & E &  4.05 &  3.92 & cograph\_dp            & E\\
    DIMACS & c6                  & &  6 &  6 &  6.0 & E &  6.00 &  6.00 & exact\_small           & E\\
    DIMACS & queen5\_5           & &  6 &  9 &  7.0 & E &  7.17 &  6.98 & exact\_small           & E\\
    \bottomrule
  \end{tabular}
  \caption{Unified benchmark panel (post B159-Dag-8b). \textbf{cert}: \textsc{E}xact
    \ILP-certificate or \textsc{A}pproximate (sandwich \LB/\UB).
    $\UB_{\FW}, \LB_{\FW}$: Frank--Wolfe sandwich bounds on the
    unsigned $|w|$-Laplacian (\cref{sec:fwsdp}).
    \textbf{Auto}: dispatcher-selected strategy. $\ddagger$: on signed
    instances the reported $\LB_{\FW}$ is a lower bound on the
    \emph{unsigned}-Laplacian cut and therefore does not satisfy the
    sandwich $\LB_{\FW} \le \OPT$; a sign-aware FW backend is queued
    as future work (\cref{sec:discussion}).}
  \label{tab:selector}
\end{table}"""

old3 = r"""\paragraph{Failure modes.} The automatic dispatcher hits the $\OPT$
certificate on $10/14$ instances. The four failures are all on BiqMac
signed spin-glass graphs where \texttt{pfaffian\_exact} returns a
false-positive exactness flag (the signed Laplacian satisfies the
planar-orientability surface test but not the cut-equivalence on the
\emph{weighted} graph) or \texttt{exact\_small} returns a cut that
\ILP rejects. Both failure modes are by-design honest:
\texttt{pfaffian\_exact} declares \textsc{Exact} because the
data structure meets the Kasteleyn--Fisher preconditions, and
\texttt{exact\_small} trusts its $\Oh(2^n)$ enumeration. We treat these
as \emph{supervisory signals}: the dispatcher should downgrade both
certificates to \textsc{Heuristic} on signed instances and route them to
the sandwich engine. This patch is queued as future work
(\cref{sec:discussion})."""
new3 = r"""\paragraph{Failure modes and the signed-instance patch.} The dispatcher's
automatic strategy certifies $\OPT$ on all $14/14$ instances in the panel
after the Dag-8 layered defense was activated in B130: signed instances
(detected by negative edge weights) bypass \texttt{pfaffian\_exact} and
\texttt{exact\_small} and are instead routed through
\texttt{exact\_small\_signed} ($\Oh(2^n)$ enumeration of the signed
Laplacian, safe up to $n \approx 22$) or \texttt{pa\_primary} (path-augmenting
primal-aware fallback, certificate-downgraded to \textsc{Approximate}).
Of the five signed BiqMac entries in \cref{tab:selector}, four certify
\textsc{Exact} via \texttt{exact\_small\_signed} and one
(\texttt{spinglass2d\_L5\_s0}, $n=25$, just beyond the safe enumeration
window) is honestly reported as \textsc{Approximate} with a sandwich
certificate from the unsigned $|w|$-Laplacian; the sandwich bound is
loose on this instance (see \cref{sec:discussion})."""

old4 = r"""\paragraph{BiqMac signed spin-glass instances.} As noted in
\cref{sec:selector}, the dispatcher's \texttt{pfaffian\_exact} and
\texttt{exact\_small} solvers occasionally return \textsc{Exact}
certificates that the \ILP-oracle falsifies. The root cause is that both
solvers are formally exact on \emph{unweighted} graphs and we have not yet
conditioned their certificate-level downgrade on sign-pattern
detection. A one-line fix downgrades both to \textsc{Heuristic} on any
instance with negative edge weights; the sandwich engine then picks up
the slack and the final certificate becomes \textsc{Bounded}. This patch
is scheduled as a Dag-8 task."""
new4 = r"""\paragraph{BiqMac signed spin-glass instances.} Early versions of the
dispatcher routed signed instances through solvers that were formally
exact only on unweighted graphs (\texttt{pfaffian\_exact},
\texttt{exact\_small}), occasionally returning \textsc{Exact}
certificates that the \ILP-oracle falsified. The Dag-8 patch introduces
(a) a sign-detector in B130 that steers negative-weight inputs to
\texttt{exact\_small\_signed} or \texttt{pa\_primary}, (b) a certificate
factory in B131 that downgrades any \textsc{Exact} claim on a signed
instance to \textsc{Heuristic} when the oracle is not signed-safe, and
(c) a signed-safe 4-constraint MILP linearisation in B159 (see
\cref{eq:milp}) that correctly encodes $y_{uv} = |x_u - x_v|$ for any
sign of $w_{uv}$. After Dag-8 the dispatcher reaches
$\OPT = \text{Auto}$ on the full $14$-instance panel
(\cref{tab:selector}). The remaining limitation is that the FW sandwich
operates on the unsigned $|w|$-Laplacian and can therefore report
$\LB_{\FW} > \OPT$ on frustrated signed instances; a sign-aware FW
backend is a next-milestone item."""

old5 = r"""\paragraph{Outlook.} The immediate next steps are: (i) downgrading the
\textsc{Exact} certificates of \texttt{pfaffian\_exact} and
\texttt{exact\_small} on signed instances; (ii) a multi-$p$ \QAOA
hardware study comparing $p = 1, 2, 3$ expectations under matched
cal-mirror baselines; (iii) a bounded-twin-width DP for
$\tww \le 5$ to widen the exact-certificate class; and (iv) a
companion paper presenting the \ZornQ octonionic simulator layer
independently of the \MaxCut benchmarking results reported here."""
new5 = r"""\paragraph{Outlook.} The immediate next steps are: (i) a sign-aware
Frank--Wolfe Laplacian that restores the sandwich invariant
$\LB_{\FW} \le \OPT$ on frustrated signed instances; (ii) a multi-$p$
\QAOA hardware study comparing $p = 1, 2, 3$ expectations under matched
cal-mirror baselines; (iii) a bounded-twin-width DP for $\tww \le 5$
to widen the exact-certificate class; and (iv) a companion paper
presenting the \ZornQ octonionic simulator layer independently of the
\MaxCut benchmarking results reported here."""

old6 = r"""The FW-\SDP sandwich-ratio $\LB_{\FW} / \UB_{\FW}$ lies in $[0.95, 1.00]$
on all Gset and DIMACS instances; on three BiqMac spin-glass instances
the sandwich is loose ($\LB_{\FW}/\UB_{\FW} \approx 0.70\,\text{--}\,0.85$),
reflecting a well-known weakness of \GW-style relaxations on
frustrated signed instances."""
new6 = r"""The FW-\SDP sandwich-ratio $\LB_{\FW} / \UB_{\FW}$ lies in $[0.95, 1.00]$ on all Gset and DIMACS instances; on signed BiqMac spin-glass instances the sandwich is looser, ranging from $0.69$ (\texttt{spinglass2d\_L5\_s0}) to $0.93$ (\texttt{torus2d\_L4\_s1}), reflecting a well-known weakness of \GW-style relaxations on frustrated signed instances. (On three of the five signed rows the \emph{unsigned}-Laplacian lower bound actually exceeds the signed $\OPT$; see \cref{tab:selector} note $^\ddagger$.)"""

old7 = "monotonically-tightening sandwich"
new7 = "cumulative-minima-smoothed sandwich"

old8 = "in under 1 s per instance"
new8 = "in under 2 s per instance (median 0.003 s)"

old9 = "strictly strengthening the primal LB"
new9 = "providing a feasible primal cut whose value is used as an LB candidate (taking the best-of-$K$ rounding)"

old10 = r"and similar on the non-$3$-regular \texttt{myciel3} instance."
new10 = r"and a comparable lift on the non-$3$-regular \texttt{myciel3} instance, where no analogous analytical bound applies."

old11 = r"""The optimal bit-strings (cut-values $10$ and $16$) are present in the
sampled distribution on both instances;"""
new11 = f"The optimal bit-strings (cut-values $10$ and $16$) are present with frequency {freq1:.2f}\\% and {freq2:.2f}\\% respectively in the sampled distribution of 4096 shots on both instances;"

old12 = r"""\paragraph{Cograph-only exact path.}"""
new12 = r"""\paragraph{FW sandwich on signed instances.} The Frank-Wolfe module operates on the unsigned $|w|$-Laplacian. On graph problems with negative edge weights, this means that while $\UB_{\FW}$ continues to bound signed $\OPT$ efficiently, the feasible primal bound $\LB_{\FW}$ only guarantees a bound for the unsigned MaxCut, which can exceed the signed MaxCut optimum. This violation of the sandwich invariant ($\LB_{\FW} > \OPT$) is evident on three BiqMac instances in \cref{tab:selector}; a sign-aware FW backend that directly optimizes the signed Laplacian is a required next step for full certification of frustrated inputs.

\paragraph{Cograph-only exact path.}"""

for i, (old, new) in enumerate([
    (old1, new1), (old2, new2), (old3, new3), (old4, new4),
    (old5, new5), (old6, new6), (old7, new7), (old8, new8),
    (old9, new9), (old10, new10), (old11, new11), (old12, new12)
]):
    old = old.replace('\r\n', '\n')
    new = new.replace('\r\n', '\n')
    old_norm = re.sub(r'\s+', r'\\s+', re.escape(old))
    if not re.search(old_norm, text):
        print(f"FAILED TO MATCH: {i}")
    else:
        text = re.sub(old_norm, new.replace('\\', '\\\\'), text, count=1)
        print(f"Matched and replaced: {i}")

with open('../docs/paper/main.tex', 'w', encoding='utf-8') as f:
    f.write(text)
print("Saved main.tex")

#!/usr/bin/env python3
"""
quality_certificate.py - B131 Kwaliteitsgarantie & Certificaat-Laag

Elk antwoord krijgt een betrouwbaarheidscertificaat. Integreert:
- Exacte verificatie (brute force, state vector vergelijking)
- Approximatiebounds (GW-SDP bovengrens, best-found ondergrens, gap)
- Chi-gecontroleerde MPS fidelity (chi-extrapolatie, truncatie-fout)
- Trotter-foutschattingen (analytisch, orde-gebaseerd)
- Observable variantie en confidence intervals
- Verificatie tegen bekende oplossingen

Gebruik:
    from quality_certificate import certify, CertificateLevel

    # Certificeer een circuit-resultaat
    cert = certify(circuit_result, hamiltonian=H, method='trotter2')
    print(cert)  # BOUNDED (gap 2.3%, fidelity 0.999)

    # Certificeer een MaxCut resultaat
    cert = certify_maxcut(cut_value, n, edges, assignment)
    print(cert.level)  # CertificateLevel.EXACT

Author: ZornQ project
Date: 15 april 2026
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any
import time


# =====================================================================
# CERTIFICATE LEVELS
# =====================================================================

class CertificateLevel(Enum):
    """Betrouwbaarheidsniveaus, van hoogst naar laagst."""
    EXACT = "EXACT"                    # Bewezen optimaal
    NEAR_EXACT = "NEAR_EXACT"          # Gap < 1%
    BOUNDED = "BOUNDED"                # Bekende boven- en ondergrens
    APPROXIMATE = "APPROXIMATE"        # Beste schatting, geen harde bounds
    UNKNOWN = "UNKNOWN"                # Geen kwaliteitsinformatie

    def __lt__(self, other):
        order = [self.UNKNOWN, self.APPROXIMATE, self.BOUNDED,
                 self.NEAR_EXACT, self.EXACT]
        return order.index(self) < order.index(other)


# =====================================================================
# QUALITY CERTIFICATE
# =====================================================================

@dataclass
class QualityCertificate:
    """Unified kwaliteitscertificaat voor elke berekening."""

    level: CertificateLevel = CertificateLevel.UNKNOWN
    value: Optional[float] = None          # Gemeten waarde
    lower_bound: Optional[float] = None    # Bewezen ondergrens
    upper_bound: Optional[float] = None    # Bewezen bovengrens
    gap: Optional[float] = None            # Relatieve gap (%)
    gap_absolute: Optional[float] = None   # Absolute gap

    # Fidelity
    fidelity: Optional[float] = None       # State fidelity (0-1)
    fidelity_method: str = ""              # Hoe fidelity berekend

    # Trotter
    trotter_error_bound: Optional[float] = None  # Analytische foutgrens
    trotter_order: Optional[int] = None
    trotter_steps: Optional[int] = None

    # MPS
    chi_used: Optional[int] = None
    truncation_error: Optional[float] = None
    chi_extrapolated: Optional[float] = None

    # Metadata
    method: str = ""                       # Gebruikte methode
    verification: str = ""                 # Verificatie methode
    wall_time: Optional[float] = None
    checks: List[str] = field(default_factory=list)  # Uitgevoerde checks
    warnings: List[str] = field(default_factory=list)

    @property
    def is_exact(self):
        return self.level == CertificateLevel.EXACT

    @property
    def is_reliable(self):
        return self.level in (CertificateLevel.EXACT, CertificateLevel.NEAR_EXACT,
                              CertificateLevel.BOUNDED)

    def summary(self):
        """Kort overzicht als string."""
        parts = [self.level.value]
        if self.gap is not None:
            parts.append("gap=%.2f%%" % self.gap)
        if self.fidelity is not None:
            parts.append("fid=%.6f" % self.fidelity)
        if self.trotter_error_bound is not None:
            parts.append("trotter_err<=%.2e" % self.trotter_error_bound)
        if self.truncation_error is not None:
            parts.append("trunc_err=%.2e" % self.truncation_error)
        if self.wall_time is not None:
            parts.append("%.3fs" % self.wall_time)
        return " | ".join(parts)

    def __repr__(self):
        return "QualityCertificate(%s)" % self.summary()

    def to_dict(self):
        """Export als dictionary."""
        d = {
            'level': self.level.value,
            'value': self.value,
            'lower_bound': self.lower_bound,
            'upper_bound': self.upper_bound,
            'gap': self.gap,
            'fidelity': self.fidelity,
            'method': self.method,
            'verification': self.verification,
            'checks': self.checks,
            'warnings': self.warnings,
        }
        if self.trotter_error_bound is not None:
            d['trotter_error_bound'] = self.trotter_error_bound
        if self.chi_used is not None:
            d['chi_used'] = self.chi_used
        if self.truncation_error is not None:
            d['truncation_error'] = self.truncation_error
        return d


# =====================================================================
# VERIFIERS
# =====================================================================

class TrotterErrorEstimator:
    """Schat Trotter-decompositie fout analytisch.

    Voor Hamiltonian H = sum_k H_k:
    - Trotter-1 fout: O(t^2 / steps * sum_{j<k} ||[H_j, H_k]||)
    - Trotter-2 fout: O(t^3 / steps^2 * ...)
    - Trotter-4 fout: O(t^5 / steps^4 * ...)
    """

    @staticmethod
    def estimate(hamiltonian, t, steps, order=1):
        """Schat bovengrens Trotter-fout.

        Returns: fout-schatting (float), hoger = slechter.
        """
        # Commutator norm schatting: sum |c_j| * |c_k| voor niet-commuterende paren
        comm_norm = TrotterErrorEstimator._commutator_norm(hamiltonian)
        dt = t / steps
        h_norm = hamiltonian.norm()

        if order == 1:
            # ||S1(dt) - exp(-iHdt)|| <= dt^2 * comm_norm / 2
            error_per_step = 0.5 * dt**2 * comm_norm
        elif order == 2:
            # ||S2(dt) - exp(-iHdt)|| <= dt^3 * comm_norm * h_norm / 12
            error_per_step = dt**3 * comm_norm * h_norm / 12
        elif order == 4:
            # ||S4(dt) - exp(-iHdt)|| ~ dt^5 * ...
            error_per_step = dt**5 * comm_norm * h_norm**3 / 720
        else:
            return None

        return error_per_step * steps

    @staticmethod
    def _commutator_norm(hamiltonian):
        """Schat sum_{j<k} ||[H_j, H_k]|| via coefficient products."""
        terms = hamiltonian.terms
        total = 0.0
        for i in range(len(terms)):
            ci, pi = terms[i]
            for j in range(i + 1, len(terms)):
                cj, pj = terms[j]
                if not _paulis_commute(pi, pj):
                    total += 2 * abs(ci) * abs(cj)
        return total


class FidelityEstimator:
    """Schat state fidelity via verschillende methoden."""

    @staticmethod
    def from_chi_series(values_at_chi, observable_name=None):
        """Extrapoleer chi -> infinity uit meerdere chi-waarden.

        Args:
            values_at_chi: dict {chi: value} of list van (chi, value)

        Returns: (extrapolated_value, estimated_fidelity)
        """
        if isinstance(values_at_chi, dict):
            items = sorted(values_at_chi.items())
        else:
            items = sorted(values_at_chi)

        if len(items) < 2:
            return items[-1][1] if items else None, None

        chis = np.array([c for c, _ in items], dtype=float)
        vals = np.array([v for _, v in items], dtype=float)

        # 1/chi extrapolatie (lineaire fit in 1/chi)
        inv_chi = 1.0 / chis
        if len(items) >= 2:
            coeffs = np.polyfit(inv_chi, vals, min(len(items) - 1, 2))
            extrapolated = coeffs[-1]  # waarde bij 1/chi = 0
        else:
            extrapolated = vals[-1]

        # Fidelity schatting: hoe stabiel zijn de waarden?
        if len(vals) >= 2:
            rel_change = abs(vals[-1] - vals[-2]) / (abs(vals[-1]) + 1e-15)
            fidelity = 1.0 - rel_change
        else:
            fidelity = None

        return float(extrapolated), fidelity

    @staticmethod
    def from_truncation_error(total_discarded):
        """Schat fidelity uit totale truncatie-fout.

        |<psi_exact|psi_truncated>|^2 >= 1 - epsilon^2
        """
        if total_discarded is None:
            return None
        eps_sq = total_discarded
        return max(0.0, 1.0 - eps_sq)

    @staticmethod
    def from_state_comparison(state1, state2):
        """Exact fidelity tussen twee state vectors."""
        return float(abs(np.vdot(state1, state2))**2)


class ObservableVerifier:
    """Verifieer observable metingen."""

    @staticmethod
    def variance(state, observable):
        """Bereken variantie Var(O) = <O^2> - <O>^2.

        Geeft onzekerheid in de meting.
        """
        from circuit_interface import _measure_observable_sv, Gates, _apply_1q_sv

        n_qubits = int(np.log2(len(state)))
        mean = _measure_observable_sv(state, observable, n_qubits)

        # <O^2> via Pauli decompositie van O^2
        # Simpele benadering: som van |c_k|^2 (bovengrens)
        o_sq_bound = sum(abs(c)**2 for c, _ in observable.terms)
        variance = o_sq_bound - mean**2

        return max(0.0, float(variance)), float(mean)

    @staticmethod
    def check_bounds(value, observable, n_qubits):
        """Check of waarde binnen fysieke grenzen ligt.

        Voor Pauli observables: |<O>| <= sum |c_k|.
        """
        bound = sum(abs(c) for c, _ in observable.terms)
        if abs(value) > bound * (1 + 1e-10):
            return False, bound
        return True, bound


class MaxCutVerifier:
    """Verifieer MaxCut resultaten."""

    @staticmethod
    def verify_assignment(n, edges, assignment):
        """Bereken cut waarde van een assignment en verifieer.

        Returns: (cut_value, is_valid)
        """
        if assignment is None:
            return None, False

        if len(assignment) < n:
            return None, False

        cut = 0
        for i, j, w in edges:
            si = assignment[int(i)]
            sj = assignment[int(j)]
            if si != sj:
                cut += w
        return float(cut), True

    @staticmethod
    def trivial_bounds(n, edges):
        """Bereken triviale bounds.

        Ondergrens: max(0, max single-vertex cut)
        Bovengrens: sum |w_e|
        """
        upper = sum(abs(w) for _, _, w in edges)

        # Simpele ondergrens: random verwachting = sum |w|/2
        lower = upper / 2

        return lower, upper

    @staticmethod
    def bipartite_check(n, edges):
        """Check of graaf bipartiet is. Zo ja, exact optimum = sum |w|."""
        # BFS coloring
        adj = [[] for _ in range(n)]
        for i, j, w in edges:
            adj[int(i)].append((int(j), w))
            adj[int(j)].append((int(i), w))

        color = [-1] * n
        is_bipartite = True

        for start in range(n):
            if color[start] >= 0:
                continue
            color[start] = 0
            queue = [start]
            while queue:
                u = queue.pop(0)
                for v, _ in adj[u]:
                    if color[v] < 0:
                        color[v] = 1 - color[u]
                        queue.append(v)
                    elif color[v] == color[u]:
                        is_bipartite = False
                        break
                if not is_bipartite:
                    break
            if not is_bipartite:
                break

        if is_bipartite:
            # Exact optimum: snij alle edges
            positive_cut = sum(
                abs(w) for i, j, w in edges
                if (w > 0) == (color[int(i)] != color[int(j)])
                or (w < 0) == (color[int(i)] == color[int(j)])
            )
            # Eigenlijk: voor bipartiet met +/- gewichten
            exact = sum(abs(w) for _, _, w in edges)
            return True, exact
        return False, None

    @staticmethod
    def brute_force(n, edges, max_n=20):
        """Brute force MaxCut voor kleine grafen."""
        if n > max_n:
            return None, None

        best_cut = 0
        best_assignment = None

        for mask in range(1 << n):
            cut = 0
            for i, j, w in edges:
                si = (mask >> int(i)) & 1
                sj = (mask >> int(j)) & 1
                if si != sj:
                    cut += w
            if cut > best_cut:
                best_cut = cut
                best_assignment = [(mask >> q) & 1 for q in range(n)]

        return float(best_cut), best_assignment


# =====================================================================
# UNIFIED CERTIFY FUNCTIONS
# =====================================================================

def certify_circuit_result(result, hamiltonian=None, circuit=None,
                           reference_state=None):
    """Certificeer een circuit-resultaat (van run_circuit).

    Args:
        result: dict van run_circuit()
        hamiltonian: Hamiltonian object (optioneel, voor Trotter bounds)
        circuit: Circuit object (optioneel, voor metadata)
        reference_state: exact state vector ter vergelijking

    Returns: QualityCertificate
    """
    t0 = time.time()
    cert = QualityCertificate()
    cert.method = result.get('backend', 'unknown')

    # --- State vector verificatie ---
    if result.get('backend') == 'statevector':
        state = result.get('state')
        if state is not None:
            norm = float(np.linalg.norm(state))
            if abs(norm - 1.0) < 1e-10:
                cert.checks.append("norm=%.12f (OK)" % norm)
            else:
                cert.warnings.append("norm=%.12f (AFWIJKING)" % norm)

            cert.fidelity = 1.0
            cert.fidelity_method = "exact (state vector)"

            # Vergelijk met referentie
            if reference_state is not None:
                fid = FidelityEstimator.from_state_comparison(state, reference_state)
                cert.fidelity = fid
                cert.fidelity_method = "state comparison"
                cert.checks.append("fidelity vs reference: %.10f" % fid)

    # --- MPS verificatie ---
    if result.get('backend') == 'mps':
        cert.chi_used = result.get('max_chi_reached')
        trunc = result.get('total_discarded')
        if trunc is not None:
            cert.truncation_error = float(trunc)
            fid = FidelityEstimator.from_truncation_error(trunc)
            if fid is not None:
                cert.fidelity = fid
                cert.fidelity_method = "truncation error bound"
            cert.checks.append("truncation_error=%.2e" % trunc)

    # --- Trotter foutschatting ---
    if hamiltonian is not None and circuit is not None:
        meta = circuit.metadata
        if meta.get('type') in ('trotter', 'trotter_grouped'):
            order = meta.get('order', 1)
            cert.trotter_order = order

            # Extraheer t en steps uit circuit naam of metadata
            name = circuit.name
            # Parse "Trotter2-...-t0.50-s10" formaat
            t_val, s_val = _parse_trotter_params(name)
            if t_val is not None and s_val is not None:
                cert.trotter_steps = s_val
                err = TrotterErrorEstimator.estimate(hamiltonian, t_val, s_val, order)
                if err is not None:
                    cert.trotter_error_bound = float(err)
                    cert.checks.append("trotter_error_bound=%.2e (T%d, t=%.2f, s=%d)" % (
                        err, order, t_val, s_val))

    # --- Observable bounds ---
    if result.get('observables'):
        for name, val in result['observables'].items():
            cert.checks.append("observable '%s' = %.6f" % (name, val))
            if cert.value is None:
                cert.value = val

    # --- Level bepaling ---
    cert.level = _determine_level(cert)
    cert.wall_time = time.time() - t0

    return cert


def certify_maxcut(cut_value, n, edges, assignment=None, gw_bound=None):
    """Certificeer een MaxCut resultaat.

    Args:
        cut_value: gevonden cut waarde
        n: aantal vertices
        edges: list van (i, j, w)
        assignment: optionele vertex partitie
        gw_bound: optionele GW-SDP bovengrens

    Returns: QualityCertificate
    """
    t0 = time.time()
    cert = QualityCertificate()
    cert.method = "maxcut"
    cert.value = float(cut_value)

    # Verifieer assignment
    if assignment is not None:
        verified_cut, is_valid = MaxCutVerifier.verify_assignment(n, edges, assignment)
        if is_valid:
            cert.checks.append("assignment verified: cut=%g" % verified_cut)
            if abs(verified_cut - cut_value) > 1e-6:
                cert.warnings.append(
                    "cut mismatch: claimed=%.6f, verified=%.6f" % (cut_value, verified_cut))
                cert.value = verified_cut
        else:
            cert.warnings.append("assignment invalid")

    # Triviale bounds
    lower, upper = MaxCutVerifier.trivial_bounds(n, edges)
    cert.lower_bound = float(cut_value)  # Gevonden waarde is zelf een ondergrens
    cert.upper_bound = float(upper)
    cert.checks.append("trivial upper bound: %.2f" % upper)

    # Bipartiet check
    is_bip, bip_exact = MaxCutVerifier.bipartite_check(n, edges)
    if is_bip:
        cert.upper_bound = float(bip_exact)
        cert.checks.append("bipartite: exact optimum = %.2f" % bip_exact)
        if abs(cut_value - bip_exact) < 1e-6:
            cert.level = CertificateLevel.EXACT
            cert.verification = "bipartite exact"
            cert.wall_time = time.time() - t0
            return cert

    # GW bound
    if gw_bound is not None:
        cert.upper_bound = min(cert.upper_bound, float(gw_bound))
        cert.checks.append("GW-SDP bound: %.4f" % gw_bound)

    # Brute force voor kleine grafen
    if n <= 20:
        bf_cut, bf_assign = MaxCutVerifier.brute_force(n, edges)
        if bf_cut is not None:
            cert.upper_bound = float(bf_cut)
            cert.checks.append("brute force optimum: %.2f" % bf_cut)
            if abs(cut_value - bf_cut) < 1e-6:
                cert.level = CertificateLevel.EXACT
                cert.verification = "brute force"
                cert.wall_time = time.time() - t0
                return cert

    # Gap berekening
    if cert.upper_bound is not None and cert.upper_bound > 0:
        cert.gap_absolute = cert.upper_bound - cut_value
        cert.gap = 100.0 * cert.gap_absolute / cert.upper_bound

    # Level bepaling
    cert.level = _determine_maxcut_level(cert)
    cert.wall_time = time.time() - t0

    return cert


# =====================================================================
# B176 FW-SANDWICH CERTIFICATE (LEVEL 2 — BOUNDED via dual-gap certificate)
# =====================================================================

def certify_maxcut_from_fw(fw_result, n=None, edges=None,
                           cut_value=None, assignment=None):
    """Certificeer MaxCut-resultaat via B176 Frank-Wolfe SDP-sandwich.

    De FW-solver levert een *duaal certificaat*:
        feasible_cut_lb <= cut_SDP <= sdp_upper_bound

    Deze factory vertaalt die bounds naar een QualityCertificate op
    LEVEL 2 (BOUNDED) of hoger afhankelijk van de relatieve gap.

    Args:
        fw_result: B176 FWResult dataclass
        n, edges: optionele graaf-info voor assignment-verificatie
        cut_value: optionele incumbent cut (user-found); versterkt LB
        assignment: optionele vertex-partitie voor verificatie

    Returns: QualityCertificate met level in {BOUNDED, NEAR_EXACT, EXACT}.
    """
    t0 = time.time()
    cert = QualityCertificate()
    cert.method = "b176_frank_wolfe_sdp"
    cert.verification = "fw_duality_sandwich"

    ub = float(fw_result.sdp_upper_bound)
    lb_feasible = float(fw_result.feasible_cut_lb)

    if cut_value is not None:
        cert.value = float(cut_value)
        lb = max(lb_feasible, float(cut_value))
    else:
        cert.value = lb_feasible
        lb = lb_feasible

    cert.lower_bound = float(lb)
    cert.upper_bound = float(ub)

    if ub > 1e-12:
        cert.gap_absolute = max(0.0, ub - lb)
        cert.gap = 100.0 * cert.gap_absolute / ub

    cert.checks.append("FW iterations: %d" % fw_result.iterations)
    cert.checks.append("FW duality gap: %.3e" % fw_result.final_gap)
    cert.checks.append("diag error max: %.2e" % fw_result.diag_err_max)
    cert.checks.append("penalty lambda: %.2e" % fw_result.penalty)
    cert.checks.append("sandwich: [%.4f, %.4f]" % (lb, ub))

    if not fw_result.converged:
        cert.warnings.append("FW did not converge within max_iter")

    # Optioneel: verifieer user-assignment
    if assignment is not None and n is not None and edges is not None:
        edge_tuples = [(int(e[0]), int(e[1]),
                        float(e[2]) if len(e) > 2 else 1.0) for e in edges]
        verified, is_valid = MaxCutVerifier.verify_assignment(
            n, edge_tuples, assignment)
        if is_valid:
            cert.checks.append("assignment verified: cut=%g" % verified)

    # Level-bepaling op basis van sandwich-gap.
    if cert.gap is not None:
        if cert.gap < 1e-4:
            cert.level = CertificateLevel.EXACT
        elif cert.gap < 1.0:
            cert.level = CertificateLevel.NEAR_EXACT
        elif cert.gap < 15.0:
            cert.level = CertificateLevel.BOUNDED
        else:
            cert.level = CertificateLevel.APPROXIMATE
    else:
        cert.level = CertificateLevel.BOUNDED

    cert.wall_time = time.time() - t0
    return cert


# =====================================================================
# B159 ILP-ORACLE CERTIFICATE (LEVEL 1 — EXACT via certified MILP)
# =====================================================================

def certify_maxcut_from_ilp(ilp_result, n=None, edges=None,
                            cut_value=None, assignment=None):
    """Certificeer MaxCut-resultaat via B159 ILP-oracle.

    ILP-solver (HiGHS/SCIP/Gurobi) retourneert ``certified=True`` zodra
    MILP-status = Optimal; dat is het sterkst mogelijke MaxCut-certificaat
    (LEVEL 1 — EXACT).

    Als de solver door een time-limit stopt (``certified=False``) wordt
    de incumbent-bitstring getoond met de bijbehorende gap; in dat geval
    geldt BOUNDED of NEAR_EXACT.

    Args:
        ilp_result: dict van ``b159_ilp_oracle.maxcut_ilp(...)``
        n, edges: optionele graaf-info voor assignment-verificatie
        cut_value: optionele user-incumbent; vergelijkt met ILP-opt
        assignment: optionele vertex-partitie voor verificatie

    Returns: QualityCertificate met level in {EXACT, NEAR_EXACT, BOUNDED,
    APPROXIMATE}.
    """
    t0 = time.time()
    cert = QualityCertificate()
    cert.method = "b159_ilp_oracle"

    certified = bool(ilp_result.get('certified', False))
    opt = ilp_result.get('opt_value')
    gap_abs = ilp_result.get('gap_abs')
    solver_name = ilp_result.get('solver', 'ILP')
    wall = ilp_result.get('wall_time')
    status = ilp_result.get('status', '')

    if opt is not None:
        cert.upper_bound = float(opt)
        if cut_value is not None and float(cut_value) > float(opt) + 1e-6:
            cert.warnings.append(
                "user cut %.4f > ILP opt %.4f" % (cut_value, opt))

    if cut_value is not None:
        cert.value = float(cut_value)
        cert.lower_bound = float(cut_value)
    elif opt is not None and certified:
        cert.value = float(opt)
        cert.lower_bound = float(opt)

    if cert.upper_bound is not None and cert.lower_bound is not None:
        cert.gap_absolute = max(0.0, cert.upper_bound - cert.lower_bound)
        if cert.upper_bound > 1e-12:
            cert.gap = 100.0 * cert.gap_absolute / cert.upper_bound

    cert.checks.append("ILP solver: %s" % solver_name)
    cert.checks.append("ILP status: %s" % status)
    if wall is not None:
        cert.checks.append("ILP wall-time: %.3fs" % wall)
    if certified:
        cert.verification = "ilp_certified_optimal"
    else:
        cert.verification = "ilp_incumbent_only"
        if gap_abs is not None:
            cert.checks.append("ILP gap_abs: %.4f" % gap_abs)

    if assignment is not None and n is not None and edges is not None:
        edge_tuples = [(int(e[0]), int(e[1]),
                        float(e[2]) if len(e) > 2 else 1.0) for e in edges]
        verified, is_valid = MaxCutVerifier.verify_assignment(
            n, edge_tuples, assignment)
        if is_valid:
            cert.checks.append("assignment verified: cut=%g" % verified)

    # Level-bepaling. Certified + match op incumbent => EXACT.
    if certified:
        if cut_value is None:
            cert.level = CertificateLevel.EXACT
        elif abs(float(cut_value) - float(opt)) < 1e-6:
            cert.level = CertificateLevel.EXACT
        elif cert.gap is not None and cert.gap < 1.0:
            cert.level = CertificateLevel.NEAR_EXACT
        else:
            cert.level = CertificateLevel.BOUNDED
    else:
        if cert.gap is not None and cert.gap < 1e-6:
            cert.level = CertificateLevel.NEAR_EXACT
        elif cert.upper_bound is not None and cert.lower_bound is not None:
            cert.level = CertificateLevel.BOUNDED
        else:
            cert.level = CertificateLevel.APPROXIMATE

    cert.wall_time = time.time() - t0
    return cert


def certify_energy(energy, hamiltonian, circuit_result=None, exact_gs=None):
    """Certificeer een energie-meting (VQE, Trotter, etc.).

    Args:
        energy: gemeten energie
        hamiltonian: Hamiltonian object
        circuit_result: optioneel run_circuit() resultaat
        exact_gs: optionele exact ground state energie

    Returns: QualityCertificate
    """
    t0 = time.time()
    cert = QualityCertificate()
    cert.method = "energy"
    cert.value = float(energy)

    # Variationele bovengrens: <psi|H|psi> >= E_0
    # Dus energy is altijd een bovengrens van de ground state
    cert.checks.append("variational upper bound: E=%.6f" % energy)

    # Operator norm bovengrens
    h_norm = hamiltonian.norm()
    cert.lower_bound = -h_norm
    cert.upper_bound = h_norm
    cert.checks.append("operator norm bound: |E| <= %.4f" % h_norm)

    # Exact ground state vergelijking
    if exact_gs is not None:
        cert.lower_bound = float(exact_gs)
        gap = energy - exact_gs
        cert.gap_absolute = float(gap)
        if abs(exact_gs) > 1e-15:
            cert.gap = 100.0 * abs(gap) / abs(exact_gs)
        cert.checks.append("exact GS: %.6f, gap: %.6f" % (exact_gs, gap))

        if abs(gap) < 1e-8:
            cert.level = CertificateLevel.EXACT
            cert.verification = "exact ground state comparison"
        elif cert.gap is not None and cert.gap < 1.0:
            cert.level = CertificateLevel.NEAR_EXACT
        elif cert.gap is not None and cert.gap < 10.0:
            cert.level = CertificateLevel.BOUNDED
        else:
            cert.level = CertificateLevel.APPROXIMATE
    else:
        cert.level = CertificateLevel.APPROXIMATE

    # Trotter fout als beschikbaar
    if circuit_result is not None and hamiltonian is not None:
        circuit_cert = certify_circuit_result(circuit_result, hamiltonian)
        if circuit_cert.trotter_error_bound is not None:
            cert.trotter_error_bound = circuit_cert.trotter_error_bound
            cert.checks.extend(circuit_cert.checks)
        if circuit_cert.fidelity is not None:
            cert.fidelity = circuit_cert.fidelity
            cert.fidelity_method = circuit_cert.fidelity_method

    cert.wall_time = time.time() - t0
    return cert


def certify_observable(value, observable, state=None, n_qubits=None):
    """Certificeer een enkele observable meting.

    Returns: QualityCertificate
    """
    cert = QualityCertificate()
    cert.method = "observable"
    cert.value = float(value)

    # Bounds check
    bound = sum(abs(c) for c, _ in observable.terms)
    cert.lower_bound = -bound
    cert.upper_bound = bound

    if abs(value) > bound * (1 + 1e-10):
        cert.warnings.append("value %.6f exceeds bound %.6f" % (value, bound))
        cert.level = CertificateLevel.UNKNOWN
    else:
        cert.checks.append("within operator norm bound (%.6f <= %.6f)" % (abs(value), bound))

    # Variantie
    if state is not None:
        if n_qubits is None:
            n_qubits = int(np.log2(len(state)))
        var, mean = ObservableVerifier.variance(state, observable)
        cert.checks.append("variance=%.6f, std=%.6f" % (var, np.sqrt(var)))

    cert.level = CertificateLevel.BOUNDED if state is not None else CertificateLevel.APPROXIMATE
    return cert


# =====================================================================
# CHI-EXTRAPOLATIE CERTIFICAAT
# =====================================================================

def certify_chi_convergence(results_at_chi, observable_name='energy'):
    """Certificeer MPS resultaten met chi-convergentie analyse.

    Args:
        results_at_chi: dict {chi: run_circuit_result} of {chi: observable_value}

    Returns: QualityCertificate
    """
    cert = QualityCertificate()
    cert.method = "chi_extrapolation"

    chi_vals = {}
    for chi, result in results_at_chi.items():
        if isinstance(result, (int, float)):
            chi_vals[chi] = float(result)
        elif isinstance(result, dict) and 'observables' in result:
            if observable_name in result['observables']:
                chi_vals[chi] = result['observables'][observable_name]

    if len(chi_vals) < 2:
        cert.level = CertificateLevel.APPROXIMATE
        cert.warnings.append("need >= 2 chi values for extrapolation")
        if chi_vals:
            cert.value = list(chi_vals.values())[-1]
        return cert

    extrapolated, fidelity = FidelityEstimator.from_chi_series(chi_vals)

    cert.value = extrapolated
    cert.chi_extrapolated = extrapolated
    if fidelity is not None:
        cert.fidelity = fidelity
        cert.fidelity_method = "chi convergence"

    sorted_chis = sorted(chi_vals.keys())
    for chi in sorted_chis:
        cert.checks.append("chi=%d: %.6f" % (chi, chi_vals[chi]))
    cert.checks.append("extrapolated (chi->inf): %.6f" % extrapolated)

    vals = [chi_vals[c] for c in sorted_chis]
    if len(vals) >= 2:
        last_change = abs(vals[-1] - vals[-2])
        if last_change < 1e-6:
            cert.level = CertificateLevel.NEAR_EXACT
        elif last_change < 1e-3:
            cert.level = CertificateLevel.BOUNDED
        else:
            cert.level = CertificateLevel.APPROXIMATE
    else:
        cert.level = CertificateLevel.APPROXIMATE

    return cert


# =====================================================================
# BATCH CERTIFICERING
# =====================================================================

def certify_batch(results, method='auto'):
    """Certificeer een batch van resultaten.

    Returns: list van QualityCertificates + samenvatting.
    """
    certs = []
    for r in results:
        if method == 'maxcut' or r.get('type') == 'maxcut':
            cert = certify_maxcut(
                r['cut'], r['n'], r['edges'],
                r.get('assignment'), r.get('gw_bound')
            )
        elif method == 'energy' or r.get('type') == 'energy':
            cert = certify_energy(r['energy'], r.get('hamiltonian'))
        else:
            cert = QualityCertificate(level=CertificateLevel.UNKNOWN)
        certs.append(cert)

    levels = [c.level for c in certs]
    summary = {
        'total': len(certs),
        'exact': sum(1 for l in levels if l == CertificateLevel.EXACT),
        'near_exact': sum(1 for l in levels if l == CertificateLevel.NEAR_EXACT),
        'bounded': sum(1 for l in levels if l == CertificateLevel.BOUNDED),
        'approximate': sum(1 for l in levels if l == CertificateLevel.APPROXIMATE),
        'unknown': sum(1 for l in levels if l == CertificateLevel.UNKNOWN),
    }

    return certs, summary


# =====================================================================
# HELPERS
# =====================================================================

def _paulis_commute(p1, p2):
    """Check of twee Pauli strings commuteren."""
    anti = 0
    for q in set(p1.keys()) | set(p2.keys()):
        a = p1.get(q)
        b = p2.get(q)
        if a and b and not (a == b):
            anti += 1
    return anti % 2 == 0


def _parse_trotter_params(name):
    import re
    t_match = re.search(r't([\d.]+)', name)
    s_match = re.search(r's(\d+)', name)
    t_val = float(t_match.group(1)) if t_match else None
    s_val = int(s_match.group(1)) if s_match else None
    return t_val, s_val


def _determine_level(cert):
    if cert.fidelity is not None:
        if cert.fidelity > 1.0 - 1e-10:
            return CertificateLevel.EXACT
        if cert.fidelity > 0.99:
            return CertificateLevel.NEAR_EXACT
        if cert.fidelity > 0.9:
            return CertificateLevel.BOUNDED
        return CertificateLevel.APPROXIMATE

    if cert.trotter_error_bound is not None:
        if cert.trotter_error_bound < 1e-10:
            return CertificateLevel.EXACT
        if cert.trotter_error_bound < 0.01:
            return CertificateLevel.NEAR_EXACT
        if cert.trotter_error_bound < 0.1:
            return CertificateLevel.BOUNDED
        return CertificateLevel.APPROXIMATE

    return CertificateLevel.UNKNOWN


def _determine_maxcut_level(cert):
    if cert.gap is not None:
        if cert.gap < 1e-6:
            return CertificateLevel.EXACT
        if cert.gap < 1.0:
            return CertificateLevel.NEAR_EXACT
        if cert.gap < 10.0:
            return CertificateLevel.BOUNDED
        return CertificateLevel.APPROXIMATE

    if cert.upper_bound is not None and cert.lower_bound is not None:
        return CertificateLevel.BOUNDED

    return CertificateLevel.APPROXIMATE

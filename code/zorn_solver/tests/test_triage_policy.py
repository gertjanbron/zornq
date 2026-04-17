from zornq_solve.policy_triage import TriagePolicy
from zornq_solve.zorn_core import Vec3, ZornState


def test_triage_accepts_regular_state():
    p = TriagePolicy(eps=1e-8)
    x = ZornState(1.0, 2.0, Vec3(1.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0))
    out = p.apply(x)
    assert out.lambda_valid is True
    assert out.state == x


def test_triage_rejects_singular_state():
    p = TriagePolicy(eps=1e-8)
    x = ZornState(1.0, 1.0, Vec3(1.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0))
    out = p.apply(x)
    assert out.lambda_valid is False
    assert out.state == ZornState.zero()

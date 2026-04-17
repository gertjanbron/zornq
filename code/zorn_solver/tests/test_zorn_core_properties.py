from zornq_solve.zorn_core import Vec3, ZornState


def test_zorn_identity():
    x = ZornState(2.0, 3.0, Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0))
    i = ZornState.identity()
    assert x.multiply(i) == x
    assert i.multiply(x) == x


def test_norm_multiplicative_on_sample():
    x = ZornState(1.0, 2.0, Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0))
    y = ZornState(3.0, 4.0, Vec3(0.0, 1.0, 0.0), Vec3(1.0, 0.0, 0.0))
    xy = x.multiply(y)
    assert abs(xy.norm() - x.norm() * y.norm()) < 1e-9

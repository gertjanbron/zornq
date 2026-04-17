#!/usr/bin/env python3
"""Debug frustrated_antiferro weight distribution."""
from adversarial_instance_generator import gen_frustrated_antiferro

inst = gen_frustrated_antiferro(n=25, p_triangle=0.4, seed=42)
weights = [w for u, v, w in inst['edges']]
pos = sum(1 for w in weights if w > 0)
neg = sum(1 for w in weights if w < 0)
print(f"pos={pos}, neg={neg}, total={sum(weights):.1f}, n_edges={len(weights)}")

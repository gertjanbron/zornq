from zornq_common.graph_types import WeightedGraph
from zornq_solve import ZornHeuristicSolver


def test_solver_returns_valid_partition():
    g = WeightedGraph.from_edges(
        num_nodes=4,
        edges=[(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0)],
    )
    solver = ZornHeuristicSolver(graph=g, time_limit_sec=0.05, max_iterations=4, seed=0)
    result = solver.solve()
    assert len(result.best_partition) == 4
    assert result.best_cut >= 0.0

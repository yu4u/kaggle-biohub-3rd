"""Keep a supplied feasible solution when a limited solve returns a worse one."""
import numpy as np


def keep_better_incumbent(problem, values, candidate, info, incumbent):
    if incumbent is None:
        return candidate, info
    old = np.asarray(incumbent, float)
    if old.shape != np.asarray(candidate).shape or not np.all((old == 0) | (old == 1)):
        return candidate, info
    advantage = float(np.dot(values, old) - np.dot(values, candidate))
    if advantage <= 1e-12 or not np.all(problem.matrix @ old <= problem.ub + 1e-6):
        return candidate, info
    return old.copy(), dict(status=info['status'], mip_gap=None,
        solver='retained better incumbent', rejected_solver=info, linear_advantage=advantage)

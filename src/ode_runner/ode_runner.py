from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from odeopt.ode import RK4
from odeopt.ode.system.nonlinearsys import BetaSEIIR


@dataclass(frozen=True)
class SiierdModelSpecs:
    alpha: float
    sigma: float
    gamma1: float
    gamma2: float
    N: float  # in case we want to do fractions, but not number of people

    def __post_init__(self):
        assert 0 < self.alpha <= 1.0
        assert self.sigma >= 0.0
        assert self.gamma1 >= 0
        assert self.gamma2 >= 0
        assert self.N > 0


class ODERunner:

    def __init__(self, model_specs: SiierdModelSpecs, init_cond: np.ndarray, dt: float):
        self.model_specs = model_specs
        self.init_cond = init_cond
        self.dt = dt

    def get_solution(self, times, beta, solver="RK4"):
        model = BetaSEIIR(**asdict(self.model_specs))
        if solver == "RK4":
            solver = RK4(model.system, self.dt)
        else:
            raise ValueError("Unknown solver type")
        solution = solver.solve(t=times, init_cond=self.init_cond, t_params=times, params=beta.reshape((1, -1)))
        result = pd.DataFrame(
            data=np.concatenate([solution, times.reshape((1, -1)), beta.reshape((1, -1))], axis=0).T,
            columns=["S", "E", "I1", "I2", "R", "t", "beta"]
        )
        return result
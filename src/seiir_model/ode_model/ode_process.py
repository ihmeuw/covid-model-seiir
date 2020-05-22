# -*- coding: utf-8 -*-
"""
    ODE Process
    ~~~~~~~~~~~~
"""
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
from odeopt.ode import RK4
from odeopt.ode import LinearFirstOrder
from odeopt.core.utils import linear_interpolate
from .spline_fit import SplineFit


class SingleGroupODEProcess:
    def __init__(self, df,
                 col_date,
                 col_cases,
                 col_pop,
                 col_loc_id,
                 today=np.datetime64(datetime.today()),
                 day_shift=(8,)*2,
                 lag_days=17,
                 alpha=(0.95,)*2,
                 sigma=(0.2,)*2,
                 gamma1=(0.5,)*2,
                 gamma2=(0.5,)*2,
                 solver_class=RK4,
                 solver_dt=1e-1,
                 spline_options=None,
                 spline_se_power=1.0):
        """Constructor of the SingleGroupODEProcess

        Args:
            df (pd.DateFrame): DataFrame contains all the data.
            col_date (str): Date of the rows.
            col_cases (str): Column with new infectious data.
            col_pop (str): Column with population.
            col_loc_id (str): Column with location id.
            today (np.datetime64): Indicating when "today" is. Defaults to the actual today.
            day_shift (arraylike): Days shift for the data sub-selection.
            alpha (arraylike): bounds for uniformly sampling alpha.
            sigma (arraylike): bounds for uniformly sampling sigma.
            gamma1 (arraylike): bounds for uniformly sampling gamma1.
            gamma2 (arraylike): bounds for uniformly sampling gamma2.
            solver_class (ODESolver, optional): Solver for the ODE system.
            solver_dt (float, optional): Step size for the ODE system.
            spline_options (dict | None, optional):
                Dictionary of spline prior options.
            splinne_se_power(float): The standard error scaling power.
        """
        # observations
        assert col_date in df
        assert col_cases in df
        assert col_pop in df
        assert col_loc_id in df
        self.col_date = col_date
        self.col_cases = col_cases
        self.col_pop = col_pop
        self.col_loc_id = col_loc_id

        self.loc_id = df[self.col_loc_id].values[0]

        # ODE parameters
        assert len(alpha) == 2 and \
               0.0 <= alpha[0] <= alpha[1]
        assert len(sigma) == 2 and \
               0.0 <= sigma[0] <= sigma[1]
        assert len(gamma1) == 2 and \
               0.0 <= gamma1[0] <= gamma1[1]
        assert len(gamma2) == 2 and \
               0.0 <= gamma2[0] <= gamma2[1]
        self.alpha = np.random.uniform(*alpha)
        self.sigma = np.random.uniform(*sigma)
        self.gamma1 = np.random.uniform(*gamma1)
        self.gamma2 = np.random.uniform(*gamma2)
        self.N = df[self.col_pop].values[0]

        assert len(day_shift) == 2 and \
            day_shift[0] <= day_shift[1]

        # subset the data
        self.today = today
        self.day_shift = int(np.random.uniform(*day_shift))
        self.lag_days = lag_days
        df.sort_values(self.col_date, inplace=True)
        date = pd.to_datetime(df[col_date])
        end_date = self.today + np.timedelta64(self.day_shift -
                                               self.lag_days, 'D')
        idx = date < end_date

        cases_threshold = 50.0
        start_date = date[df[col_cases] >= cases_threshold].min()
        idx_final = idx & (date >= start_date)
        while not any(idx_final):
            cases_threshold *= 0.5
            print(f'reduce cases threshold for {self.loc_id} to'
                  f'{cases_threshold}')
            start_date = date[df[col_cases] >= cases_threshold].min()
            idx_final = idx & (date >= start_date)
            if cases_threshold < 1e-6:
                break
        if np.sum(idx_final) < 2:
            cases_threshold = 0.0
            start_date = date[df[col_cases] > cases_threshold].min()
            idx_final = idx & (date > start_date)

        assert np.sum(idx_final) > 2, \
            f'loc_id: {self.loc_id}, not enough non-zero cases data to fit a ' \
            f'spline. Number of data between date {start_date} and {end_date}' \
            f' is {np.sum(idx_final)}.'

        self.df = df[idx_final].copy()
        date = date[idx_final]

        # compute days
        self.col_days = 'days'
        self.df[self.col_days] = (date - date.min()).dt.days.values

        # parse input
        self.date = self.df[self.col_date]
        self.t = self.df[self.col_days].values
        self.obs = self.df[self.col_cases].values
        self.init_cond = {
            'S': self.N,
            'E': self.obs[0],
            'I1': 1.0,
            'I2': 0.0,
            'R': 0.0
        }

        # ode solver setup
        self.solver_class = solver_class
        self.solver_dt = solver_dt
        self.t_span = np.array([np.min(self.t), np.max(self.t)])
        self.t_params = np.arange(self.t_span[0],
                                  self.t_span[1] + self.solver_dt,
                                  self.solver_dt)
        self.params = None
        self.components = None
        self.create_ode_sys()

        # spline solver setup
        self.spline_options = {
            'spline_knots': np.linspace(0.0, 1.0, 7),
            'spline_degree': 3,
            'prior_spline_convexity': 'concave',
        }
        self.spline_se_power = spline_se_power
        if spline_options is not None:
            self.spline_options.update(**spline_options)
        self.create_spline()

    def create_spline(self):
        """Create spline fit object.
        """
        self.step_spline_model = SplineFit(
            self.t[self.obs > 0.0],
            np.log(self.obs[self.obs > 0.0]),
            self.spline_options,
            se_power=self.spline_se_power
        )

    def create_ode_sys(self):
        """Create 1D ODE solver.
        """
        self.step_ode_sys = LinearFirstOrder(
            self.sigma,
            solver_class=self.solver_class,
            solver_dt=self.solver_dt
        )

    def update_ode_params(self,
                          alpha=None,
                          sigma=None,
                          gamma1=None,
                          gamma2=None):
        """Update given parameters.

        Args:
            alpha (float | None, optional):
                Updated alpha parameter, if `None` no update will happen.
            sigma (float | None, optional):
                Updated sigma parameter, if `None` no update will happen.
            gamma1 (float | None, optional):
                Updated gamma1 parameter, if `None` no update will happen.
            gamma2 (float | None, optional):
                Updated gamma2 parameter, if `None` no update will happen.
        """
        if alpha is not None:
            assert 0.0 < alpha <= 1.0
            self.alpha = alpha
        if sigma is not None:
            assert sigma >= 0.0
            self.sigma = sigma
        if gamma1 is not None:
            assert gamma1 >= 0.0
            self.gamma1 = gamma1
        if gamma2 is not None:
            assert gamma2 >= 0.0
            self.gamma2 = gamma2

    def update_data(self, obs):
        """Update data.

        Args:
            obs (np.ndarray): independent variable.
        """
        assert self.obs.size == obs.size
        self.obs = obs
        self.create_spline()

    def fit_spline(self):
        """Fit spline.
        """
        self.step_spline_model.fit_spline()
        rhs_newE = np.exp(self.step_spline_model.predict(self.t_params))
        rhs_newE[self.t_params > self.step_spline_model.spline.knots[-1]] = 0.0
        self.rhs_newE = rhs_newE

    def process(self, fit_spline=True):
        """Process the data.
        """
        # fit the spline and predict the right-hand-side
        if fit_spline:
            self.fit_spline()

        # fit the E
        self.step_ode_sys.update_given_params(c=self.sigma)
        E = self.step_ode_sys.simulate(self.t_params,
                                       np.array([self.init_cond['E']]),
                                       self.t_params,
                                       self.rhs_newE[None, :])[0]

        # fit I1
        self.step_ode_sys.update_given_params(c=self.gamma1)
        # modify initial condition of I1
        self.init_cond.update({
            'I1': (self.rhs_newE[0]/5.0)**(1.0/self.alpha)
        })
        I1 = self.step_ode_sys.simulate(self.t_params,
                                        np.array([self.init_cond['I1']]),
                                        self.t_params,
                                        self.sigma*E[None, :])[0]

        # fit I2
        self.step_ode_sys.update_given_params(c=self.gamma2)
        I2 = self.step_ode_sys.simulate(self.t_params,
                                        np.array([self.init_cond['I2']]),
                                        self.t_params,
                                        self.gamma1*I1[None, :])[0]

        # fit S
        self.init_cond.update({
            'S': self.N - self.init_cond['E'] - self.init_cond['I1']
        })
        self.step_ode_sys.update_given_params(c=0.0)
        S = self.step_ode_sys.simulate(self.t_params,
                                       np.array([self.init_cond['S']]),
                                       self.t_params,
                                       -self.rhs_newE[None, :])[0]
        neg_S_idx = S < 0.0

        # fit R
        self.step_ode_sys.update_given_params(c=0.0)
        R = self.step_ode_sys.simulate(self.t_params,
                                       np.array([self.init_cond['R']]),
                                       self.t_params,
                                       self.gamma2*I2[None, :])[0]

        if np.any(neg_S_idx):
            id_min = np.min(np.arange(S.size)[neg_S_idx])
            S[id_min:] = S[id_min - 1]
            E[id_min:] = E[id_min - 1]
            I1[id_min:] = I1[id_min - 1]
            I2[id_min:] = I2[id_min - 1]
            R[id_min:] = R[id_min - 1]

        self.components = {
            'S': S,
            'E': E,
            'I1': I1,
            'I2': I2,
            'R': R
        }

        # get beta
        self.params = (self.rhs_newE/
                       ((S/self.N)*
                        ((I1 + I2)**self.alpha)))[None, :]

    def predict(self, t):
        params = linear_interpolate(t, self.t_params, self.params)
        components = {
            c: linear_interpolate(t, self.t_params, self.components[c])
            for c in self.components
        }
        components.update({
            'newE': linear_interpolate(t, self.t_params, self.rhs_newE)
        })
        components.update({
            'newE_obs': linear_interpolate(t, self.t, self.obs)
        })
        return params, components

    def create_result_df(self):
        """Create result DataFrame.
        """
        params, components = self.predict(self.t)
        df_result = pd.DataFrame({
            'loc_id': self.loc_id,
            'date': self.date,
            'days': self.t,
            'beta': params[0]
        })

        for k, v in components.items():
            df_result[k] = v

        return df_result

    def create_params_df(self):
        """Create parameter DataFrame.
        """
        df_params = pd.DataFrame([self.alpha, self.sigma,
                                  self.gamma1, self.gamma2],
                                 index=['alpha', 'sigma', 'gamma1', 'gamma2'],
                                 columns=['params'])

        return df_params


@dataclass
class ODEProcessInput:
    df_dict: Dict
    col_date: str
    col_cases: str
    col_pop: str
    col_loc_id: str
    col_lag_days: str
    col_observed: str

    alpha: Tuple
    sigma: Tuple
    gamma1: Tuple
    gamma2: Tuple
    solver_dt: float
    spline_options: Dict
    day_shift: Tuple
    spline_se_power: float = 1.0


class ODEProcess:
    """ODE Process for multiple group.
    """
    def __init__(self, input):
        """Constructor of ODEProcess.

        Args:
            input: ODEProcessInput
        """
        self.df_dict = input.df_dict
        self.col_date = input.col_date
        self.col_cases = input.col_cases
        self.col_pop = input.col_pop
        self.col_loc_id = input.col_loc_id
        self.col_lag_days = input.col_lag_days
        self.col_observed = input.col_observed

        self.solver_dt = input.solver_dt
        self.spline_options = input.spline_options
        self.spline_se_power = input.spline_se_power

        # create the location id
        self.loc_ids = np.sort(list(self.df_dict.keys()))

        # sampling the parameters here
        self.alpha = np.random.uniform(*input.alpha)
        self.sigma = np.random.uniform(*input.sigma)
        self.gamma1 = np.random.uniform(*input.gamma1)
        self.gamma2 = np.random.uniform(*input.gamma2)
        self.day_shift = int(np.random.uniform(*input.day_shift))

        # lag days
        self.lag_days = self.df_dict[self.loc_ids[0]][
            self.col_lag_days].values[0]
        self.today_dict = {
            loc_id: np.datetime64(
                self.df_dict[loc_id].loc[self.df_dict[loc_id][self.col_observed] == 1, self.col_date].max()
            )
            for loc_id in self.loc_ids
        }

        # create model for each location
        self.models = {}
        errors = []
        for loc_id in self.loc_ids:
            try:
                self.models[loc_id] = SingleGroupODEProcess(
                    self.df_dict[loc_id],
                    self.col_date,
                    self.col_cases,
                    self.col_pop,
                    self.col_loc_id,
                    day_shift=(self.day_shift,)*2,
                    lag_days=self.lag_days,
                    alpha=(self.alpha,)*2,
                    sigma=(self.sigma,)*2,
                    gamma1=(self.gamma1,)*2,
                    gamma2=(self.gamma2,)*2,
                    solver_class=RK4,
                    solver_dt=self.solver_dt,
                    spline_options=self.spline_options,
                    today=self.today_dict[loc_id],
                    spline_se_power=self.spline_se_power
                )
            except AssertionError:
                errors.append(loc_id)

        if errors:
            raise RuntimeError(
                "SingleGroupODEProcess failed to initialize for 1 or more locations in "
                f"ODEProcess. Locations are: {errors}."
            )

    def process(self):
        """Process all models.
        """
        for loc_id, model in self.models.items():
            model.process()

    def create_result_df(self):
        """Create result DataFrame.
        """
        return pd.concat([
            model.create_result_df()
            for loc_id, model in self.models.items()
        ])

    def create_params_df(self):
        """Create parameter DataFrame.
        """
        df_params = pd.DataFrame({
            'params': ['alpha', 'sigma', 'gamma1', 'gamma2', 'day_shift'],
            'values': [self.alpha, self.sigma, self.gamma1, self.gamma2, self.day_shift]
        })

        return df_params

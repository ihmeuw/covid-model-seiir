# -*- coding: utf -*-
"""
    Simple spline fitting class using mrbrt.
"""
import numpy as np
import pandas as pd
from mrtool import MRData
from mrtool import LinearCovModel
from mrtool import MRBRT


class SplineFit:
    """Spline fit class
    """
    def __init__(self, t, y,
                 spline_options=None,
                 se_power=1.0,
                 space='ln daily',
                 max_iter=50):
        """Constructor of the SplineFit

        Args:
            t (np.ndarray): Independent variable.
            y (np.ndarray): Dependent variable.
            spline_options (dict | None, optional):
                Dictionary of spline prior options.
            se_power (float):
                A number between 0 and 1 that scale the standard error.
            space (str):
                Which space is the spline fitting, assume y is daily cases.
            max_iter (int):
                Maximum number of iteration.
        """
        self.space = space
        assert self.space in ['daily', 'ln daily', 'cumul', 'ln cumul'], "spline_space must be one of 'daily'," \
                                                                         " 'ln daily', 'cumul', 'ln cumul' space."
        if self.space == 'ln daily':
            self.t = t[y > 0.0]
            self.y = np.log(y[y > 0.0])
        elif self.space == 'daily':
            self.t = t
            self.y = y
        elif self.space == 'ln cumul':
            y = np.cumsum(y)
            self.t = t[y > 0.0]
            self.y = np.log(y[y > 0.0])
        else:
            self.t = t
            self.y = np.cumsum(y)
        self.spline_options = {} if spline_options is None else spline_options
        self.se_power = se_power

        assert 0 <= self.se_power <= 1, "spline se_power has to be between 0 and 1."
        if self.se_power == 0:
            y_se = np.ones(self.t.size)
        else:
            y_se = 1.0/np.exp(self.y)**self.se_power
        # create mrbrt object
        df = pd.DataFrame({
            'y': self.y,
            'y_se': y_se,
            't': self.t,
            'study_id': 1,
        })

        data = MRData(
            df=df,
            col_obs='y',
            col_obs_se='y_se',
            col_covs=['t'],
            col_study_id='study_id',
            add_intercept=True
        )

        intercept = LinearCovModel(
            alt_cov='intercept',
            use_re=True,
            prior_gamma_uniform=np.array([0.0, 0.0]),
            name='intercept'
        )

        time = LinearCovModel(
            alt_cov='t',
            use_re=False,
            use_spline=True,
            **self.spline_options,
            name='time'
        )

        self.mr_model = MRBRT(data, cov_models=[intercept, time])
        self.spline = time.create_spline(data)
        self.spline_coef = None
        self.max_iter = max_iter
    
    def fit_spline(self):
        """Fit the spline.
        """
        self.mr_model.fit_model(inner_max_iter=self.max_iter)
        self.spline_coef = self.mr_model.beta_soln
        self.spline_coef[1:] += self.spline_coef[0]

    def predict(self, t):
        """Predict the dependent variable, given independent variable.
        """
        mat = self.spline.design_mat(t)
        y = mat.dot(self.spline_coef)

        if self.space == 'ln daily':
            return np.exp(y)
        elif self.space == 'daily':
            return y
        elif self.space == 'ln cumul':
            y = np.exp(y)
            return np.insert(np.diff(y) / np.diff(t), 0, y[0])
        elif self.space == 'cumul':
            return np.insert(np.diff(y) / np.diff(t), 0, y[0])

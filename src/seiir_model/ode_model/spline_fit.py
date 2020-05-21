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
                 se_space='linear'):
        """Constructor of the SplineFit

        Args:
            t (np.ndarray): Independent variable.
            y (np.ndarray): Dependent variable.
            spline_options (dict | None, optional):
                Dictionary of spline prior options.
            se_space (str):
                If 'linear' assume have same standard error in the linear space,
                if 'log' assume have same standard error in the log space.
        """
        self.t = t
        self.y = y
        self.spline_options = {} if spline_options is None else spline_options
        self.se_space = se_space

        assert self.se_space in ['linear', 'log'], "spline se_space has to be linear or log."
        if self.se_space == 'linear':
            y_se = 1.0/np.exp(self.y)
        else:
            y_se = np.ones(self.y.size)
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

    def fit_spline(self):
        """Fit the spline.
        """
        self.mr_model.fit_model(inner_max_iter=30)
        self.spline_coef = self.mr_model.beta_soln
        self.spline_coef[1:] += self.spline_coef[0]

    def predict(self, t):
        """Predict the dependent variable, given independent variable.
        """
        mat = self.spline.design_mat(t)
        return mat.dot(self.spline_coef)

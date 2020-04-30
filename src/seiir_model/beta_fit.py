import pandas as pd
import numpy as np
import copy

from slime.core import MRData
from slime.model import CovModelSet, MRModel


class BetaRegressor:

    def __init__(self, covmodel_set, mr_data):
        self.covmodel_set = covmodel_set
        self.mr_data = mr_data

    def set_fe(self):
        covmodel_set_fe = copy.deepcopy(self.covmodel_set)
        for covmodel in covmodel_set_fe.cov_models:
            covmodel.use_re = False 

        mr_model_fe = MRModel(self.mr_data, covmodel_set_fe)
        mr_model_fe.fit_model()
        fe = list(mr_model_fe.result.values())[0]
        for v, covmodel in zip(fe, self.covmodel_set.cov_models):
            covmodel.bounds = np.array([v, v])

    def fit(self, two_stage=False):
        if two_stage:
            self.set_fe()
        
        self.mr_model = MRModel(self.mr_data, self.covmodel_set) 
        self.mr_model.fit_model()
        self.cov_coef = self.mr_model.result

    def predict(self, cov, group):
        if group in self.cov_coef:
            assert cov.shape[1] == len(self.cov_coef[group])
            return np.sum([self.cov_coef[group][i] * cov[:, i] for i in range(cov.shape[1])], axis=0)
        else:
            raise RuntimeError('Group Not Found.')


        
        

        


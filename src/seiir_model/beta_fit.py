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
        for covmodel in covmodel_set_fe:
            covmodel.use_re = False 

        mr_model_fe = MRModel(self.mr_data, covmodel_set_fe)
        mr_model_fe.fit_model()
        fe = list(mr_model_fe.result)[0]

        for i, covmodel in enumerate(self.covmodel_set):
            covmodel.bounds = np.array([fe[i], fe[i]])

    def fit(self, two_stage=False):
        if two_stage:
            self.set_fe()
        
        self.mr_model = MRModel(self.mr_data, self.covmodel_set) 
        self.mr_model.fit_model()
        self.cov_coef = self.mr_model.result

    def predict(self, cov):
        return self.covmodel_set.predict(cov)


        
        

        


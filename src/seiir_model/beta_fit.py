import pandas as pd
import numpy as np
import copy

from slime.core import MRData
from slime.model import CovModelSet, MRModel


class BetaRegressor:

    def __init__(self, covmodel_set, mr_data):
        self.covmodel_set = covmodel_set
        self.mr_data = mr_data
        self.col_covs = [covmodel.col_cov for covmodel in covmodel_set.cov_models]

    def set_fe(self, std):
        covmodel_set_fe = copy.deepcopy(self.covmodel_set)
        for covmodel in covmodel_set_fe.cov_models:
            covmodel.use_re = False 

        mr_model_fe = MRModel(self.mr_data, covmodel_set_fe)
        mr_model_fe.fit_model()
        fe = list(mr_model_fe.result.values())[0]
        for v, covmodel in zip(fe, self.covmodel_set.cov_models):
            covmodel.gprior = np.array([v, std])

    def fit(self, two_stage=False, std=None):
        if two_stage:
            assert std is not None
            self.set_fe(std)
        
        self.mr_model = MRModel(self.mr_data, self.covmodel_set) 
        self.mr_model.fit_model()
        self.cov_coef = self.mr_model.result

    def save_coef(self, path):
        df = pd.DataFrame.from_dict(self.cov_coef, orient='index', columns=self.col_covs)
        return df.to_csv(path)

    def load_coef(self, path):
        cov_coef_dict = pd.read_csv(path).to_dict(orient='index')
        self.cov_coef = {}
        for k, v in cov_coef_dict.items():
            coef = [v[cov] for cov in self.col_covs]
            self.cov_coef[k] = coef 

    def predict(self, cov, group):
        if group in self.cov_coef:
            assert cov.shape[1] == len(self.cov_coef[group])
            return np.sum([self.cov_coef[group][i] * cov[:, i] for i in range(cov.shape[1])], axis=0)
        else:
            raise RuntimeError('Group Not Found.')


def predict(regressor, df_cov, col_t, col_group, col_scenario):
    df = df_cov.sort_values(by=[col_group, col_scenario, col_t])
    groups = df[col_group].unique()
    col_covs = regressor.col_covs

    beta_pred = []
    
    for group in groups:
        df_one_group = df[df[col_group] == group]
        scenarios = df_one_group[col_scenario].unique()
        for sce in scenarios:
            df_one_sce = df_one_group[df_one_group[col_scenario] == sce][col_covs]
            cov = df_one_sce[col_covs].to_numpy()
            betas = regressor.predict(cov, group)
            beta_pred.append(betas)
    beta_pred = np.concatenate(beta_pred)
    df['beta'] = beta_pred

    return df

        
        

        


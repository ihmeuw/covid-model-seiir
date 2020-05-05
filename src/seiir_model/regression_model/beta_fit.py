import pandas as pd
import numpy as np
import copy

from slime.core import MRData
from slime.model import CovModelSet, MRModel, CovModel


class BetaRegressor:

    def __init__(self, covmodel_set):
        self.covmodel_set = covmodel_set
        self.col_covs = [covmodel.col_cov for covmodel in covmodel_set.cov_models]

    def fit_no_random(self, mr_data):
        self.covmodel_set_fixed = copy.deepcopy(self.covmodel_set)
        for covmodel in self.covmodel_set_fixed.cov_models:
            covmodel.use_re = False 
            covmodel.gprior = None

        self.mr_model_fixed = MRModel(mr_data, self.covmodel_set_fixed)
        self.mr_model_fixed.fit_model()
        # cov_coef_fixed = list(self.mr_model_fixed.result.values())
        # for coef in cov_coef_fixed[1:]:
        #     assert np.linalg.norm(coef - cov_coef_fixed[0]) < 1e-10
        self.cov_coef_fixed = list(self.mr_model_fixed.result.values())[0]

    def fit(self, mr_data):        
        self.mr_model = MRModel(mr_data, self.covmodel_set) 
        self.mr_model.fit_model()
        self.cov_coef = self.mr_model.result

    def save_coef(self, path):
        df = pd.DataFrame.from_dict(self.cov_coef, orient='index')
        df.reset_index(inplace=True)
        df.columns = ['group_id'] + self.col_covs
        return df.to_csv(path)

    def load_coef(self, df=None, path=None):
        if df is None:
            assert path is not None
            df = pd.read_csv(path)
        assert 'group_id' in df
        cov_coef_dict = df.set_index('group_id').to_dict(orient='index')
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


class BetaRegressorSequential:

    def __init__(self, ordered_covmodel_sets, std=None):
        if std is not None:
            assert len(ordered_covmodel_sets) == len(std)
            assert all([s > 0.0 for s in std])
            self.std = copy.deepcopy(std)
        else:
            self.std = [-1.0] * len(ordered_covmodel_sets)
        
        self.ordered_covmodel_sets = copy.deepcopy(ordered_covmodel_sets)
        self.col_covs = []
        for covmodel_set in self.ordered_covmodel_sets:
            self.col_covs.extend([covmodel.col_cov for covmodel in covmodel_set.cov_models])
    
    def fit(self, mr_data, verbose=False, add_intercept=True):
        if add_intercept:
            covmodels = [CovModel(col_cov='intercept', use_re=True, re_var=np.inf)]
            self.col_covs.insert(0, 'intercept')
        else:
            covmodels = []
        
        original_covmodels = copy.deepcopy(covmodels)
        while len(self.ordered_covmodel_sets) > 0:
            new_cov_models = self.ordered_covmodel_sets.pop(0).cov_models
            original_covmodels.extend(copy.deepcopy(new_cov_models))
            covmodel_set = CovModelSet(covmodels + new_cov_models)
            
            regressor = BetaRegressor(covmodel_set)
            regressor.fit_no_random(mr_data)
            if verbose:
                print(regressor.cov_coef_fixed)

            std = self.std.pop(0)
            for covmodel, coef in zip(covmodel_set.cov_models[len(covmodels):], regressor.cov_coef_fixed[len(covmodels):]):
                if std > 0.0:
                    covmodel.gprior = [coef, std]
                else:
                    covmodel.bounds = np.array([coef, coef])
            covmodels = covmodel_set.cov_models
        self.regressor = BetaRegressor(CovModelSet(original_covmodels))
        self.regressor.fit(mr_data)
        self.cov_coef = self.regressor.cov_coef

    def save_coef(self, path):
        self.regressor.save_coef(path)

    def load_coef(self, df=None, path=None):
        self.regressor.load_coef(df=df, path=path)

    def predict(self, cov, group):
        return self.regressor.predict(cov, group)


def predict(regressor, df_cov, col_t, col_group, col_beta='beta_pred'):
    df = df_cov.sort_values(by=[col_group, col_t])
    df['intercept'] = 1.0
    groups = df[col_group].unique()
    col_covs = regressor.col_covs

    beta_pred = []
    
    for group in groups:
        df_one_group = df[df[col_group] == group]
        if group in regressor.cov_coef:
            cov = df_one_group[col_covs].to_numpy()
            betas = regressor.predict(cov, group)
            beta_pred.append(betas)
        else:
            beta_pred.append([np.nan] * df_one_group.shape[0])
    
    beta_pred = np.concatenate(beta_pred)
    df[col_beta] = beta_pred

    return df

        
        

        


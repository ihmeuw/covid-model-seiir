import pandas as pd

from slime.core.data import MRData

COL_T = 'days'
COL_BETA = 'beta'
COL_GROUP = 'loc_id'

def convert_inputs_for_beta_model(data_cov, df_beta, covmodel_set):
    df_cov, col_t_cov, col_group_cov = data_cov
    df = df_beta.merge(
        df_cov, 
        left_on=[COL_T, COL_GROUP], 
        right_on=[col_t_cov, col_group_cov],
    ).copy()
    df.sort_values(inplace=True, by=[COL_GROUP, COL_T])
    cov_names = [covmodel.col_cov for covmodel in covmodel_set.cov_models]
    mrdata = MRData(df, col_group=COL_GROUP, col_obs=COL_BETA, col_covs=cov_names)

    return mrdata
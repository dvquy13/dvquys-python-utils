import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing as sklearn_preprocessing
import logging

from functools import wraps
from IPython.display import display

from lib.dvquys_python_utils.pandas import utils as pandas_utils

class ZscoreOutlierHandler():
    @staticmethod
    def convert_zscore(df: pd.DataFrame, cols: list = None):
        scaled = df.select_dtypes(include='number').astype(np.float64)
        if scaled.shape[1] == 0:
            logging.error("No numeric columns to convert")
            return df
        ss = sklearn_preprocessing.StandardScaler()
        if cols is None:
            return pd.DataFrame(ss.fit_transform(scaled), columns=scaled.columns, index=scaled.index)
        else:
            scaled.loc[:, cols] = ss.fit_transform(scaled[cols])
            return pd.DataFrame(scaled, columns=scaled.columns, index=scaled.index)

    @staticmethod
    @pandas_utils.get_num_diff_rows(remove_kind='[outliers by zscore]')
    def remove_outliers(
        df: pd.DataFrame,
        cols: list = None,
        upper_bound: int = 3,
        return_original_df: bool = False,
    ) -> pd.DataFrame:
        scaled = ZscoreOutlierHandler.convert_zscore(df.copy(), cols)
        if return_original_df:
            res = df.copy()
        else:
            res = scaled
        if cols is None:
            return res.loc[
                (
                    np.abs(scaled) <= upper_bound
                ).all(axis=1)
            ]
        else:
            return res.loc[
                (
                    np.abs(scaled[cols]) <= upper_bound
                ).all(axis=1)
            ]

class IQROutlier():
    @staticmethod
    def remove_outliers(q_arr):
        q1 = q_arr.quantile(0.25)
        q3 = q_arr.quantile(0.75)
        iqr = q3 - q1
        return (q_arr >= q1 - 1.5 * iqr) & (q_arr <= q3 + 1.5 * iqr)
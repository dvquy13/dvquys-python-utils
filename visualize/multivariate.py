import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
plt.style.use('seaborn-ticks')

from lib.dvquys_python_utils.pandas import outlier as outlier_utils

class PairViz():
    @staticmethod
    def pairplot(data: pd.DataFrame, cols: list, remove_outlier: bool = True, hue: str = None, **kwargs):
        if remove_outlier:
            plot_data = outlier_utils.ZscoreOutlierHandler.remove_outliers(data, cols, return_original_df=True)
        else:
            plot_data = data
        sns.pairplot(
            plot_data,
            vars=cols,
            hue=hue,
            palette="muted",
            plot_kws={"s": 15},
            diag_kind='kde',
            height=4,
            **kwargs
        )
        plt.show()
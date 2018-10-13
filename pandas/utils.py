import pandas as pd
import numpy as np

def summarize_df(
    count_missing=True,
    count_outliers=True,
    zscore_upperbound=3
):
    def _summarize_df(method):
        @wraps(method)
        def summarize(*args, **kw):
            df = method(*args, **kw)
            if kw.__contains__('activate_summarize_df_deco') and not kw['activate_summarize_df_deco']:
                return df
            display(df.head())
            display(df.shape)
            summary_df = pd.DataFrame(index=df.columns, data=np.nan, columns=['missing_count', 'missing_perc', 'outlier_count', 'outlier_perc'])
            summary_df['dtype'] = df.dtypes.apply(str)
            if count_missing:
                summary_df = summary_df.assign(
                    missing_count = df.isnull().sum(axis=0),
                    missing_perc = lambda s_df: (
                        s_df['missing_count']
                        /
                        df.shape[0]
                    )   
                )
            if count_outliers and df.select_dtypes(include='number').shape[1] > 0:
                scaled = ZscoreOutlierHandler.convert_zscore(df)
                summary_df = summary_df.assign(

                    outlier_count = scaled.where(
                        np.abs(scaled) > zscore_upperbound
                    ).count(axis=0),

                    outlier_perc = lambda df: (
                        df['outlier_count']
                        /
                        scaled.notnull().sum(axis=0)
                    )
                )
            logging.info("Summary:")
            display(
                summary_df
                .replace(
                    np.nan, 0
                )
                .loc[
                    lambda df:
                    ~(
                        (df['missing_count'] == 0)
                        &
                        (df['outlier_count'] == 0)
                    )
                ].sort_values(
                    by=['missing_perc', 'outlier_perc'], ascending=[False, False]
                ).style.format(
                    formatter=dict(zip(
                        [col for col in summary_df.columns if col.endswith('_perc')],
                        [r'{:.2%}'.format for x in range(sum(summary_df.columns.str.endswith('_perc')))]
                    ))
                )
                
            )
            return df
        return summarize
    return _summarize_df

def get_num_diff_rows(remove_kind='rows'):
    def _calc_diff(method):
        @wraps(method)
        def _calc(*args, **kw):
            old_df = args[0]
            result_df = method(*args, **kw)
            num_outliers_removed = old_df.shape[0] - result_df.shape[0]
            print(f"{num_outliers_removed} ({num_outliers_removed / old_df.shape[0]:.2%}) {remove_kind} removed")
            return result_df
        return _calc
    return _calc_diff
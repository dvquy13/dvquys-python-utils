from IPython.display import display
import sys

def previewDf(df):
    display(df.head())
    display(df.shape)

def get_variables_memSize():
    def sizeof_fmt(num, suffix='B'):
        ''' By Fred Cirera, after https://stackoverflow.com/a/1094933/1870254'''
        for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
            if abs(num) < 1024.0:
                return "%3.1f%s%s" % (num, unit, suffix)
            num /= 1024.0
        return "%.1f%s%s" % (num, 'Yi', suffix)

    for name, size in sorted(((name, sys.getsizeof(value)) for name,value in locals().items()),
                             key= lambda x: -x[1])[:10]:
        print("{:>30}: {:>8}".format(name,sizeof_fmt(size)))


def summarize_df(
    count_missing=True,
    count_outliers=True,
    zscore_upperbound=3
):
    """
    A decorator use to display summary about missing and outlier status of a DataFrame
    """
    def _summarize_df(method):
        @wraps(method)
        def summarize(*args, **kw):
            df = method(*args, **kw)
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
            print("Summary:")
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
import pandas as pd

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
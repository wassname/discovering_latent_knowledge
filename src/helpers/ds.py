from datasets import Dataset

def ds_keep_cols(ds: Dataset, cols: list) -> Dataset:
    cols_all = set(ds.features.keys())
    cols_drop = cols_all-set(cols)
    return ds.remove_columns(cols_drop)

import pandas as pd
import pyarrow.parquet as pq

def removebadval(data, badval=999999):
  return data.replace(badval, pd.NA)

def skipbadrow(data, badthres=0.5):
  na_threshold = int(badthres * len(data.columns))
  return data.dropna(thresh=na_threshold)

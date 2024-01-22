import pandas as pd
import pyarrow.parquet as pq

def removebadval(data, badval=999999):
  return data.replace(badval, pd.NA)

def skipbadrow(data, badthres=0.5):
  na_threshold = int(badthres * len(data.columns))
  return data.dropna(thresh=na_threshold)

def cleanQs(data, yidx):
  return data[data['Q'+str(yidx)] >= 0.9999]

def extract(data, yidx, features = ['X'+str(i+1) for i in range(375)]):
  data = cleanQs(data, yidx)
  return data['Y'+str(yidx)], data[features]

def extract_both(data, features = ['X'+str(i+1) for i in range(375)]):
  data = cleanQs(data, 1)
  data = cleanQs(data, 2)
  return data[['Y1', 'Y2']], data[features]
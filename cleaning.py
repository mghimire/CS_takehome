import pandas as pd
import pyarrow.parquet as pq

def removebadval(data, badval=999999):
  return data.replace(badval, pd.NA)

def skipbadrow(data, badthres=0.5):
  na_threshold = int(badthres * len(data.columns))
  return data.dropna(thresh=na_threshold)

def cleanQs(data, yidx):
  return data[data['Q'+str(yidx)] >= 0.9999]

def extract(data, yidx, features=[['X375', 'X128', 'X119', 'X120', 'X121', 'X122', 'X123', 'X124', 'X125', 'X126'],
                                  ['X324', 'X117', 'X304', 'X26', 'X25']]):
  data = cleanQs(data, yidx)
  return data['Y'+str(yidx)], data[features[yidx-1]], data['time']
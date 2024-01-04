import numpy as np
import pyarrow.parquet as pq

def removebadval(data, badval=999999):
  data = np.where(data != badval, data, NaN)

def skipbadrow(data, badthres=0.7):
  for i, dat in enumerate(data):
    if len(np.argwhere(dat == NaN)) > badthres*375:
      data = np.delete(data, i, axis=0)

if __name__ == "__main__":
  
  
      

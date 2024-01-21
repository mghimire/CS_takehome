import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from dataimport import import_csv_parquet

if __name__ == "__main__":
    filename = "../QR_TAKEHOME_20220103.csv.parquet"
    df = import_csv_parquet(filename)

    # Use pandas plotting extension
    df.plot(x='time', y='X25', kind='line')
    plt.show()
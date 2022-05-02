#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from glob import glob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str, help='Root directory of data')
    parser.add_argument('-t', '--type', type=str, default='i32', help='Data type')
    args = parser.parse_args()

    files = glob(f'{args.dir}/{args.type}/*.csv')
    files.sort()

    N = [int(f.split('/')[-1].rstrip('.csv')) for f in files]
    data = [pd.read_csv(f, delimiter='\t+', index_col=False, engine='python') for f in files]
    nrows = len(data[0].index)
    for row in range(nrows):
        times = [d.iloc[row, 3] for d in data]
        plt.loglog(N, times, label=f'{data[0].iloc[row, 0].rstrip()}: {data[0].iloc[row, 1].rstrip()} scan. {data[0].iloc[row, 2].rstrip()} search.')
    plt.legend()
    plt.xlabel('N')
    plt.ylabel('Time (ns)')
    plt.show()



if __name__ == '__main__':
    main()

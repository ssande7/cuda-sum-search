#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from glob import glob
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str, help='Root directory of data')
    parser.add_argument('-t', '--type', type=str, default='i32', help='Data type')
    parser.add_argument('-x', '--exclude', type=int, nargs='+', default=[], help='Rows to exclude')
    parser.add_argument('-l', '--linear', action='store_true', help='Linear scale axes')
    parser.add_argument('-y', '--ymax', type=float, default=None, help='Maximum for y axis')
    parser.add_argument('-o', '--output', type=str, help='output file')
    args = parser.parse_args()

    files = glob(f'{args.dir}/{args.type}/*.csv')
    files.sort()

    N = [int(f.split('/')[-1].rstrip('.csv')) for f in files]
    data = [pd.read_csv(f, delimiter='\t+', index_col=False, engine='python') for f in files]
    nrows = len(data[0].index)
    plt.figure()
    ax = plt.axes()
    if not args.linear:
        ax.set_xscale('log')
        ax.set_yscale('log')
    colours = plt.cm.Paired(np.linspace(0,1,12))
    print(colours)
    colours = colours[[0, 1, 2, 3, 11, 5, 7, 8, 9],:]
    for row in range(nrows):
        if row in args.exclude:
            # dummy plot to keep colours consistent
            ax.plot(np.nan, np.nan, label='_')
            continue
        times = np.array([d.iloc[row, 3] for d in data])
        err = np.array([d.iloc[row, 4] for d in data]) # standard deviation as error bar
        ax.errorbar(N, times,
                    yerr=err,
                    fmt='.-',
                    label=f'{data[0].iloc[row, 0].rstrip()}: {data[0].iloc[row, 1].rstrip()}. {data[0].iloc[row, 2].rstrip()} search.'
                            .replace(': Linear', ': Linear scan')
                            .replace('efficient', 'efficient scan')
                            .replace('Partial', 'Partial scan'),
                    color=colours[row,:]
                    )
    if args.ymax:
        plt.ylim(top=args.ymax, bottom=0)
    # xmark = 1024
    # _, xmax = plt.xlim()
    # ymin,ymax=plt.ylim()
    # while xmark < xmax:
    #     ax.plot([xmark, xmark], [ymin, ymax], 'k--', label='_')
    #     xmark *= 1024
    ax.legend()
    ax.set_xlabel('N')
    ax.set_ylabel('Time (ns)')
    ax.set_title(f'Sum and search time for {args.type}')
    plt.grid()
    if args.output:
        plt.savefig(args.output)
    else:
        plt.show()



if __name__ == '__main__':
    main()

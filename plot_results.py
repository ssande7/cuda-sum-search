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
    parser.add_argument('-b', '--block-size', type=int, default=128, help='Block size')
    parser.add_argument('-x', '--exclude', type=int, nargs='+', default=[], help='Rows to exclude')
    parser.add_argument('-l', '--linear', action='store_true', help='Linear scale axes')
    parser.add_argument('-y', '--ymax', type=float, default=None, help='Maximum for y axis')
    parser.add_argument('--xmin', type=float, default=None, help='Minimum for x axis')
    parser.add_argument('-o', '--output', type=str, help='output file')
    parser.add_argument('--throughput', action='store_true', help='Y axis is throughput (time per item)')
    parser.add_argument('--hide-conflicts', action='store_true', help='Hide "with conflicts" in legend')
    parser.add_argument('--show-search', action='store_true', help='Show search type in legend')
    args = parser.parse_args()

    files = glob(f'{args.dir}/{args.block_size}/{args.type}/*.csv')
    files.sort()

    N = np.array([int(f.split('/')[-1].rstrip('.csv')) for f in files])
    data = [pd.read_csv(f, delimiter='\t+', index_col=False, engine='python') for f in files]
    nrows = len(data[0].index)
    plt.figure()
    ax = plt.axes()
    if not args.linear:
        ax.set_xscale('log')
        if not args.throughput:
            ax.set_yscale('log')
    colours = plt.cm.Paired(np.linspace(0,1,12))
    colours = colours[[0, 0, 1, 1, 8, 5, 5, 5, 3, 3, 2, 9, 9],:]
    formats = ['.--', '.-', '.--', '.-', '.-', '.:', '.--', '.-', '.--', '.-', '.-', '.--', '.-']
    for row in range(nrows):
        if row in args.exclude:
            # dummy plot to keep colours consistent
            ax.plot(np.nan, np.nan, label='_')
            continue
        times = np.array([d.iloc[row, 3] for d in data], dtype=np.float64)
        err = np.array([d.iloc[row, 4] for d in data], dtype=np.float64) # standard deviation as error bar
        if args.throughput:
            err /= times
            times = N / times
            err *= times
        label = f'{data[0].iloc[row, 0].rstrip()}: {data[0].iloc[row, 1].rstrip()}.'
        if args.show_search:
            label += f' {data[0].iloc[row, 2].rstrip()} search.'
        label = label.replace(': Linear', ': Linear scan').replace('efficient', 'efficient scan')
        label = label.replace('Partial', 'Partial scan').replace('GPU Binary', 'Binary')
        if args.hide_conflicts:
            label = label.replace(' with conflicts', '')
        ax.errorbar(N, times,
                    yerr=err,
                    fmt=formats[row], color=colours[row,:],
                    label=label,
                    )
    if args.ymax:
        plt.ylim(top=args.ymax, bottom=0)
    if args.xmin:
        plt.xlim(left=args.xmin)

    ax.legend(title=f'Block Size: {args.block_size}',
              title_fontproperties={'weight': 'bold'})
    ax.set_xlabel('N')
    if args.throughput:
        ax.set_ylabel('Throughput (items per ns)')
    else:
        ax.set_ylabel('Time (ns)')
    ax.set_title(f'Sum and search time for {args.type}')
    plt.grid()
    if args.output:
        plt.savefig(args.output)
    else:
        plt.show()

if __name__ == '__main__':
    main()

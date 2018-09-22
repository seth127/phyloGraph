import phyloGraph as ph

import pandas as pd
import numpy as np
import argparse
from time import time
from datetime import datetime

parser = argparse.ArgumentParser()

parser.add_argument("--root", type=int, help="id to start from")
parser.add_argument("--name", type=str, default='name', help="name of organism to start with (for file naming).")
parser.add_argument("--max_raw", type=int, default=1000000, help="max number of raw lines to fetch from Tree of Life")
parser.add_argument("--keep_text", action="store_true", help="whether to also write out the free text of the infobox.")
parser.add_argument("--from_noage", action="store_true", help="optionally skip the tree of life phase by passing a noage file")

args = parser.parse_args()

# load data object
pgd = ph.phyloData()

# get noage data
noage_name = "data/{}-{}-noage.csv".format(args.name, args.root)
if args.from_noage:
    pgd.df = pd.read_csv(noage_name)
    print("loaded from {}".format(noage_name))
    print(pgd.df.shape)
else:
    # fetch from Tree of Life
    start = time()
    print("Fetching from Tree of Life :: {}".format(datetime.now()))
    pgd.fetch_tol_data(args.root, limit=args.max_raw)
    print("Done fetching in {} secs".format(np.round(time() - start, 0)))

    pgd.df.to_csv(noage_name, index=False)
    print("saved to {}".format(noage_name))

# get age from wikipedia
print("Getting age :: {}".format(datetime.now()))
start = time()
write_name = "data/{}-{}-df.csv".format(args.name, args.root)
print("writing to {}".format(write_name))
pgd.add_time(write=write_name, keep_text=args.keep_text)
print("Done getting age in {} secs".format(np.round(time() - start, 0)))

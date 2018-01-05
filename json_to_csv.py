import json
import csv

import argparse

parser = argparse.ArgumentParser(description='JSOn to CSV')
parser.add_argument('--json_path', dest='path_json',  default='tune_results.json')
parser.add_argument('--csv_path', dest='path_csv', default='tune_results.csv')

args = parser.parse_args()

with open(args.path_json, 'r') as f:
    j = json.load(f)
    sorted_j = sorted(j, key = lambda res: res[4])

    with open(args.path_csv, 'w') as f2:
        c = csv.writer(f2)
        c.writerow(["mesh_x","mesh_y","lm_alpha", "lm_beta", "wer"])

        for res in sorted_j:
            c.writerow([res[0], res[1], res[2], res[3], res[4]])


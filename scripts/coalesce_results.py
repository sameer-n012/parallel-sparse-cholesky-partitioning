import json
import glob
import sys
import os
from collections import defaultdict
import time


def main():

    results_dir = "results"
    files = glob.glob(os.path.join(results_dir, "results_*.json"))

    if not files:
        print(f"No results_*.json files found in '{results_dir}'")
        sys.exit(1)

    all_records = []

    for f in sorted(files):
        with open(f) as fh:
            data = json.load(fh)
        for rec in data:
            all_records.append(rec)

    def make_hashable(obj):
        if isinstance(obj, dict):
            return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
        elif isinstance(obj, list):
            return tuple(make_hashable(x) for x in obj)
        elif isinstance(obj, set):
            return tuple(sorted(make_hashable(x) for x in obj))
        else:
            return obj

    def remove_duplicates(dict_list):
        seen = set()
        result = []
        
        for d in dict_list:
            h = make_hashable(d)
            if h not in seen:
                seen.add(h)
                result.append(d)
        
        return result

    all_records = remove_duplicates(all_records)

    new_fname = f"results_{time.time()}.json"
    with open(os.path.join(results_dir, new_fname), "w", encoding="utf-8") as f:
        f.write(json.dumps(all_records, indent=2, sort_keys=True) + "\n")

    print(f"Coalesced {len(files)} results files into {new_fname}")


if __name__ == "__main__":
    main()
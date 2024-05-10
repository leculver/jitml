#!/usr/bin/python
"""Train a classification and regression model to predict CSE effectiveness."""

import argparse
import os
from typing import Dict

from pandas import DataFrame

from jitml import SuperPmi, SuperPmiCache, MethodKind, MethodContext, CseCandidate, JitType
from train import validate_core_root

def parse_args():
    """usage:  classification.py [-h] [--core_root CORE_ROOT] [--parallel n] mch"""
    parser = argparse.ArgumentParser()
    parser.add_argument("mch", help="The mch file of functions to train on.")
    parser.add_argument("--core_root", default=None, help="The coreclr root directory.")

    args = parser.parse_args()
    args.core_root = validate_core_root(args.core_root)
    return args

def write_data(data : Dict[str, object], no_cse_method : MethodContext, heuristic_method : MethodContext,
               cse_candidate : CseCandidate, perf_score : float):
    """Builds the data dictionary for a dataframe."""

    assert no_cse_method.index == heuristic_method.index
    data["method"] = no_cse_method.index
    data["cse_index"] = cse_candidate.index

    data["no_cse_score"] = no_cse_method.perf_score
    data["heuristic_score"] = heuristic_method.perf_score
    data["cse_score"] = perf_score
    data["heuristic_selected"] = cse_candidate.index - 1 in heuristic_method.cses_chosen
    data.update(cse_candidate.model_dump())
    if "type" in data:
        data["type"] = JitType(data["type"])

def to_dataframe(mch, core_root) -> DataFrame:
    """Converts the SuperPmi data to a DataFrame."""
    failed = {}
    applied = {}

    cache = SuperPmiCache(mch, core_root)
    with SuperPmi(mch, core_root) as superpmi:
        cse_perfscores = cache.get_cse_perfscores(superpmi)

        # applicable
        for method, scores in cse_perfscores.items():
            no_cse_method = cache.jit_method(superpmi, method, MethodKind.NO_CSE)
            heuristic_method = cache.jit_method(superpmi, method, MethodKind.HEURISTIC)
            if not no_cse_method or not heuristic_method:
                print(f"Skipping {method} due to missing methods.")
                continue

            for i, score in enumerate(scores):
                cse_candidate = no_cse_method.cse_candidates[i]
                cache.jit_method(superpmi, method, [cse_candidate.index])

                if cse_candidate.viable:
                    if score is not None:
                        write_data(applied, no_cse_method, heuristic_method, cse_candidate, score)
                    else:
                        write_data(failed, no_cse_method, heuristic_method, cse_candidate, score)

    return DataFrame(applied)

def get_or_create_dataframe(mch, core_root) -> DataFrame:
    filename = f"{mch}.cse.csv"
    if os.path.exists(filename):
        return DataFrame.read_csv(filename)

    df = to_dataframe(mch, core_root)
    df.to_csv(filename)
    return df

def main(args):
    """Main entry point."""
    if not SuperPmiCache.exists(args.mch):
        print(f"Caching SuperPmi methods for {args.mch}, this may take several minutes...")

    df = get_or_create_dataframe(args.mch, args.core_root)


if __name__ == '__main__':
    main(parse_args())

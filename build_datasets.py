#!/usr/bin/python

from concurrent.futures import ThreadPoolExecutor
import hashlib
import itertools
import math
import os
import argparse
from queue import Queue
from threading import Lock
from typing import Iterable, List
import pandas as pd
import numpy as np
from tqdm import tqdm

from jitml import SuperPmiContext, MethodContext
from train import validate_core_root

def get_no_cse(ctx : SuperPmiContext):
    with ctx.create_superpmi() as spmi:
        result = []
        for m in tqdm(spmi.enumerate_methods(JitMetrics=1, JitRLHook=1, JitRLHookCSEDecisions=[]), unit="methods"):
            result.append(m)

        return result



def get_heuristic(ctx : SuperPmiContext):
    with ctx.create_superpmi() as spmi:
        return list(spmi.enumerate_methods(JitMetrics=1))

def create_initial_data(output_root : str, ctx : SuperPmiContext) -> pd.DataFrame:
    methods_file = output_root + ".methods.parquet"
    perf_scores_file = output_root + ".perf_scores.parquet"
    cse_candidates_file = output_root + ".cse_candidates.parquet"

    if os.path.exists(methods_file) and os.path.exists(perf_scores_file) and os.path.exists(cse_candidates_file):
        methods = pd.read_parquet(methods_file, engine="pyarrow")
        perf_scores = pd.read_parquet(perf_scores_file, engine="pyarrow")
        cse_candidates = pd.read_parquet(cse_candidates_file, engine="pyarrow")

        return methods, perf_scores, cse_candidates

    print("JIT'ing methods without CSEs, this will take several minutes...")
    with ThreadPoolExecutor() as executor:
        heuristic = executor.submit(get_heuristic, ctx)
        no_cse = executor.submit(get_no_cse, ctx)

        heuristic = { x.index : x for x in heuristic.result() }
        no_cse = no_cse.result()

    seen = set()

    methods = {}
    perf_scores = {}
    cse_candidates = {}

    for method in tqdm(no_cse, desc="Processing methods...", unit="methods"):
        if method.index not in heuristic:
            continue

        viable = [x.index for x in method.cse_candidates if x.viable]
        if not viable:
            continue

        cse_hash = np.stack([c.to_hashable_tensor() for c in method.cse_candidates], axis=0).tobytes()
        cse_hash = hashlib.sha256(cse_hash).hexdigest()
        if cse_hash in seen:
            continue

        seen.add(cse_hash)

        methods.setdefault("id", []).append(method.index)
        methods.setdefault("name", []).append(method.name)
        methods.setdefault("hash", []).append(cse_hash)
        methods.setdefault("num_viable_cses", []).append(len(viable))
        methods.setdefault("viable_cses", []).append(viable)
        methods.setdefault("enreg_count", []).append(method.cse_candidates[0].enreg_count)

        for cse in method.cse_candidates:
            cse_candidates.setdefault("method_id", []).append(method.index)
            cse_candidates.setdefault("cse_index", []).append(cse.index)
            cse_candidates.setdefault("live_across_call", []).append(cse.live_across_call)
            cse_candidates.setdefault("const", []).append(cse.const)
            cse_candidates.setdefault("shared_const", []).append(cse.shared_const)
            cse_candidates.setdefault("make_cse", []).append(cse.make_cse)
            cse_candidates.setdefault("has_call", []).append(cse.has_call)
            cse_candidates.setdefault("containable", []).append(cse.containable)
            cse_candidates.setdefault("type", []).append(cse.type)
            cse_candidates.setdefault("cost_ex", []).append(cse.cost_ex)
            cse_candidates.setdefault("cost_sz", []).append(cse.cost_sz)
            cse_candidates.setdefault("use_count", []).append(cse.use_count)
            cse_candidates.setdefault("def_count", []).append(cse.def_count)
            cse_candidates.setdefault("use_wt_cnt", []).append(cse.use_wt_cnt)
            cse_candidates.setdefault("def_wt_cnt", []).append(cse.def_wt_cnt)
            cse_candidates.setdefault("distinct_locals", []).append(cse.distinct_locals)
            cse_candidates.setdefault("local_occurrences", []).append(cse.local_occurrences)
            cse_candidates.setdefault("bb_count", []).append(cse.bb_count)
            cse_candidates.setdefault("block_spread", []).append(cse.block_spread)

        perf_scores.setdefault("method_id", []).append(method.index)
        perf_scores.setdefault("cses_chosen", []).append(bytes([]))
        perf_scores.setdefault("perf_score", []).append(method.perf_score)
        perf_scores.setdefault("total_bytes", []).append(method.total_bytes)
        perf_scores.setdefault("instruction_count", []).append(method.instruction_count)

        heuristic_method = heuristic[method.index]
        perf_scores.setdefault("method_id", []).append(method.index)
        perf_scores.setdefault("cses_chosen", []).append(bytes([255]))
        perf_scores.setdefault("perf_score", []).append(heuristic_method.perf_score)
        perf_scores.setdefault("total_bytes", []).append(heuristic_method.total_bytes)
        perf_scores.setdefault("instruction_count", []).append(heuristic_method.instruction_count)

    methods = pd.DataFrame(methods)
    methods.set_index("id", inplace=True)
    methods.to_parquet(methods_file, engine="pyarrow", compression="zstd")

    cse_candidates = pd.DataFrame(cse_candidates)
    cse_candidates.set_index(["method_id", "cse_index"], inplace=True)
    cse_candidates.to_parquet(cse_candidates_file, engine="pyarrow", compression="zstd")

    perf_scores = pd.DataFrame(perf_scores)
    perf_scores.set_index(["method_id", "cses_chosen"], inplace=True)
    perf_scores.to_parquet(perf_scores_file, engine="pyarrow", compression="zstd")

    return methods, perf_scores, cse_candidates

def generate_combinations(n : int, k : int, count : int) -> Iterable[List[int]]:
    total = math.comb(n, k)
    if total <= 40_000:
        result = list(itertools.combinations(range(n), k))
        if len(result) < count:
            return result

        np.random.shuffle(result)
        return result[:count]

    result = set()
    while len(result) < count:
        result.add(tuple(sorted(np.random.choice(n, k, replace=False))))

    return result

def jit_cses_worker(ctx : SuperPmiContext, methods : pd.DataFrame, curr : int, progress : tqdm):
    perf_scores = {}
    with ctx.create_superpmi() as spmi:
        for method_id, method in methods.iterrows():
            for cses_chosen in itertools.combinations(method['viable_cses'], curr):
                progress.update(1)
                result = spmi.jit_method(method_id, JitRLHook=1, JitRLHookCSEDecisions=cses_chosen)
                if not result:
                    continue

                add_perf_score(perf_scores, cses_chosen, result)


    perf_scores = pd.DataFrame(perf_scores)
    return perf_scores

def add_perf_score(dictionary, cses_chosen, result : MethodContext):
    dictionary.setdefault("method_id", []).append(result.index)
    dictionary.setdefault("cses_chosen", []).append(bytes(cses_chosen))
    dictionary.setdefault("perf_score", []).append(result.perf_score)
    dictionary.setdefault("total_bytes", []).append(result.total_bytes)
    dictionary.setdefault("instruction_count", []).append(result.instruction_count)

def jit_cses(output_root : str, ctx : SuperPmiContext, methods : pd.DataFrame,
             count : int, parallelism : int) -> pd.DataFrame:
    file = output_root + f".perf_scores_{count}.parquet"
    if os.path.exists(file):
        return pd.read_parquet(file, engine="pyarrow")

    methods = methods[methods.num_viable_cses >= count]

    total = sum(math.comb(row['num_viable_cses'], count) for _, row in methods.iterrows())
    progress = tqdm(desc=f"JIT'ing {count} len CSEs", total=total)
    try:
        parts = np.array_split(methods, parallelism)
        with ThreadPoolExecutor(parallelism) as executor:
            futures = []
            for part in parts:
                futures.append(executor.submit(jit_cses_worker, ctx, part, count, progress))

            for future in futures:
                combined = [future.result() for future in futures]

            perf_scores = pd.concat(combined)
    finally:
        progress.close()

    perf_scores.set_index(["method_id", "cses_chosen"], inplace=True)
    perf_scores.to_parquet(file, engine="pyarrow", compression="zstd")
    return perf_scores

DATA_POINTS = 8192

def jit_partial_cses_worker(ctx : SuperPmiContext, methods : pd.DataFrame, prev_methods : pd.DataFrame,
                                count : int, total : int, progress : tqdm):
    prev_perf_scores = {}
    perf_scores = {}
    with ctx.create_superpmi() as spmi:
        for idx, row in methods[methods.num_viable_cses >= count].iterrows():
            viable = row['viable_cses']
            for combination in generate_combinations(len(viable), count, 1 + total // count):
                cses_chosen = [viable[x] for x in combination]
                method = spmi.jit_method(idx, JitRLHook=1, JitRLHookCSEDecisions=cses_chosen)

                if method:
                    add_perf_score(perf_scores, cses_chosen, method)
                    for i, _ in enumerate(cses_chosen):
                        curr = cses_chosen[:i] + cses_chosen[i+1:]
                        key = (idx, bytes(curr))
                        if key not in prev_methods:
                            continue

                        prev = spmi.jit_method(idx, JitRLHook=1, JitRLHookCSEDecisions=curr)
                        if prev:
                            add_perf_score(prev_perf_scores, curr, prev)

            progress.update(1)

    prev_perf_scores = pd.DataFrame(prev_perf_scores)
    perf_scores = pd.DataFrame(perf_scores)

    return prev_perf_scores, perf_scores

def jit_partial_cses(ctx : SuperPmiContext, methods : pd.DataFrame, prev_methods : pd.DataFrame,
                        count : int, total : int, parallelism : int) -> pd.DataFrame:

    methods = methods[methods.num_viable_cses >= count]

    prev_perf_scores = []
    perf_scores = []

    parts = np.array_split(methods, parallelism)
    with ThreadPoolExecutor(parallelism) as executor:
        futures = []
        progress = tqdm(desc=f"JIT'ing {count} len CSEs", total=len(methods))
        for part in parts:
            futures.append(executor.submit(jit_partial_cses_worker, ctx, part, prev_methods, count, total, progress))

        for future in futures:
            prev, curr = future.result()
            prev_perf_scores.append(prev)
            perf_scores.append(curr)

    prev_perf_scores = pd.concat(prev_perf_scores, ignore_index=True)
    perf_scores = pd.concat(perf_scores, ignore_index=True)

    if len(prev_perf_scores) > 0:
        prev_perf_scores.set_index(["method_id", "cses_chosen"], inplace=True)

    perf_scores.set_index(["method_id", "cses_chosen"], inplace=True)

    return prev_perf_scores, perf_scores

def main(args):
    os.makedirs(args.output, exist_ok=True)
    output_root = os.path.join(args.output, os.path.splitext(os.path.basename(args.mch))[0])

    ctx = SuperPmiContext(mch=args.mch, core_root=args.core_root)

    methods, _, _ = create_initial_data(output_root, ctx)

    # Calculate all single, double, triple cses
    for i in range(1, 4):
        jit_cses(output_root, ctx, methods, i, args.parallel)

    # Calculate enough data points for the partial CSEs
    prev_scores = None
    for i in range(4, 32):
        prev_file = output_root + f".perf_scores_{i-1}.parquet"
        file = output_root + f".perf_scores_{i}.parquet"
        if os.path.exists(file):
            prev_scores = None
            prev_file = file
            continue

        if prev_scores is None:
            prev_scores = pd.read_parquet(prev_file, engine="pyarrow")

        updated_scores, scores = jit_partial_cses(ctx, methods, prev_scores, i, DATA_POINTS, args.parallel)
        if len(updated_scores) > 0:
            prev_scores = pd.concat([prev_scores, updated_scores], ignore_index=True)
            prev_scores.set_index(["method_id", "cses_chosen"], inplace=True)
            prev_scores.to_parquet(prev_file, engine="pyarrow", compression="zstd")

        scores.to_parquet(file, engine="pyarrow", compression="zstd")
        prev_scores = scores

def parse_args():
    """usage:  build_datasets.py [-h] [--core_root CORE_ROOT] mch"""
    parser = argparse.ArgumentParser()
    parser.add_argument("mch", help="The mch file of functions to train on.")
    parser.add_argument("--core_root", default=None, help="The coreclr root directory.")
    parser.add_argument("--output", default=None, help="The output directory to save the datasets to.")
    parser.add_argument("--parallel", type=int, default=8, help="The number of parallel JITs to run.")

    args = parser.parse_args()
    args.core_root = validate_core_root(args.core_root)

    if args.output is None:
        # current file's path
        args.output = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")

    return args

if __name__ == "__main__":
    main(parse_args())

"""This builds a DataFrame from the SuperPmi data for individual CSE decisions.  Essentially we will JIT every possible
CSE decision and compare the performance of the JITted code to the original JITted code.  We will then build a
DataFrame with this data.

The DataFrame will have the following columns:
- method: The method index.
- cse_index: The CSE index.
- no_cse_score: The performance score of the method without CSE.
- heuristic_score: The performance score of the method with the current CSE's chosen by heuristic.
- cse_score: The performance score of the method with only the current CSE applied.
- heuristic_selected: True if the heuristic selected this CSE.
- All CseCandidate fields.
"""

import os
from typing import Dict, List

from pandas import DataFrame, read_csv

from .superpmi import SuperPmiCache, SuperPmi, MethodKind
from .method_context import CseCandidate, JitType, MethodContext


def update_data(data : Dict[str, List[object]], no_cse_method : MethodContext, heuristic_method : MethodContext,
               cse_candidate : CseCandidate, perf_score : float):
    """Builds the data dictionary for a dataframe."""

    assert no_cse_method.index == heuristic_method.index
    data.setdefault("method", []).append(no_cse_method.index)
    data.setdefault("cse_index", []).append(cse_candidate.index)
    data.setdefault("no_cse_score", []).append(no_cse_method.perf_score)
    data.setdefault("heuristic_score", []).append(heuristic_method.perf_score)
    data.setdefault("cse_score", []).append(perf_score)
    data.setdefault("heuristic_selected", []).append(cse_candidate.index - 1 in heuristic_method.cses_chosen)
    for key, value in cse_candidate.model_dump().items():
        if key == "type":
            value = JitType(value)

        data.setdefault(key, []).append(value)

def to_dataframe(mch, core_root) -> DataFrame:
    """Converts the SuperPmi data to a DataFrame."""
    failed = {}
    applied = {}

    cache = SuperPmiCache(mch, core_root)
    with SuperPmi(mch, core_root) as superpmi:
        cse_perfscores = cache.get_cse_perfscores(superpmi)

        # applicable
        for method, scores in cse_perfscores.items():
            method = int(method)
            no_cse_method = cache.jit_method(superpmi, method, MethodKind.NO_CSE)
            heuristic_method = cache.jit_method(superpmi, method, MethodKind.HEURISTIC)
            if not no_cse_method or not heuristic_method:
                print(f"Skipping {method} due to missing methods.")
                continue

            for i, score in enumerate(scores):
                cse_candidate = no_cse_method.cse_candidates[i]
                if cse_candidate.viable:
                    if score is not None:
                        update_data(applied, no_cse_method, heuristic_method, cse_candidate, score)
                    else:
                        update_data(failed, no_cse_method, heuristic_method, cse_candidate, score)

    return DataFrame.from_dict(applied)

def get_individual_cse_perf(mch, core_root) -> DataFrame:
    """Gets or creates the DataFrame for the given mch file."""
    filename = f"{mch}.cse.csv"
    if os.path.exists(filename):
        return read_csv(filename, index_col=0)

    df = to_dataframe(mch, core_root)
    df.to_csv(filename)
    return df

__all__ = [get_individual_cse_perf.__name__]

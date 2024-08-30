"""A wrapper around the CSE_Candidate and MethodContext classes.  CseCandidate mirrors code in
src/coreclr/jit/optcse.cpp."""

from enum import Enum
from typing import List, Optional
import numpy as np
from pydantic import BaseModel, ValidationError, field_validator

class JitType(Enum):
    """The type of a CSE candidate.  Mirrors CSE_HeuristicRLHook's enum."""
    OTHER : int = 0
    INT : int = 1
    LONG : int = 2
    FLOAT : int = 3
    DOUBLE : int = 4
    STRUCT : int = 5
    SIMD : int = 6

class CseCandidate(BaseModel):
    """A CSE candidate.  Mirrors CSE_Candidate features in CSE_HeuristicRLHook.cpp."""
    index : int
    applied : Optional[bool] = False
    viable : bool
    live_across_call : bool
    const : bool
    shared_const : bool
    make_cse : bool
    has_call : bool
    containable : bool
    type : int
    cost_ex : int
    cost_sz : int
    use_count : int
    def_count : int
    use_wt_cnt : int
    def_wt_cnt : int
    distinct_locals : int
    local_occurrences : int
    bb_count : int
    block_spread : int
    enreg_count : int

    @field_validator('applied', 'viable', 'live_across_call', 'const', 'shared_const', 'make_cse', 'has_call',
                     'containable', mode='before')
    @classmethod
    def validate_bool(cls, v):
        """Validates that the value is a boolean or is a 0 or 1."""
        if isinstance(v, int) and v in [0, 1]:
            return bool(v)

        if isinstance(v, bool):
            return v

        raise ValidationError(f"Value must be either 1, 0, or a boolean, got {v}")

    @property
    def can_apply(self):
        """Returns True if the candidate is viable and not applied."""
        return self.viable and not self.applied

    def to_hashable_tensor(self, tolerance=1e-5):
        """Returns a hashable tensor of the CSE candidate.  This is used for hashing and comparison."""
        tensor = np.array([self.viable, self.live_across_call, self.const, self.shared_const, self.make_cse,
                           self.has_call, self.containable, self.type, self.cost_ex, self.cost_sz, self.use_count,
                           self.def_count, self.use_wt_cnt, self.def_wt_cnt, self.distinct_locals,
                           self.local_occurrences, self.bb_count, self.block_spread, self.enreg_count
                           ], dtype=np.float32)

        # round to the nearest tolerance
        return np.round(tensor / tolerance) * tolerance


class MethodContext(BaseModel):
    """A superpmi method context."""
    index : int
    name : str
    hash : str
    total_bytes : int
    prolog_size : int
    instruction_count : int
    perf_score : float
    bytes_allocated : int
    num_cse : int
    num_cse_candidate : int
    heuristic : str
    cses_chosen : List[int]
    cse_candidates : List[CseCandidate]

    def __str__(self):
        return f"{self.index}: {self.name}"

    # validate that perf_score is never negative:
    @field_validator('perf_score', mode='before')
    @classmethod
    def _validate_perf_score(cls, v):
        if v < 0:
            raise ValueError("perf_score must not be negative")
        return v

class CSEDecision(BaseModel):
    """A common format for storing the outcome of choosing a specific CSE decision."""
    method : MethodContext
    heuristic_perfscore : float
    no_cse_perfscore : float
    cse_perfscore : List[float | None]

__all__ = [
    CseCandidate.__name__,
    MethodContext.__name__,
    JitType.__name__
]

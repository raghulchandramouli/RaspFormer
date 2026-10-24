"""InterpBench subset: six RASP programs with SIIT-trained transformers and known circuits."""

from bench.programs import (CASES, BenchCase, compile_case, evaluate,
                            reference_outputs, sample_inputs)

__all__ = ["CASES", "BenchCase", "compile_case", "evaluate", "reference_outputs",
           "sample_inputs"]

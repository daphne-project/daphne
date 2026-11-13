import subprocess
import re
import csv
import os
import itertools
from typing import Literal

# configuration
DAPHNE        = "./bin/daphne"
SCRIPT_DIR    = "/tmp/daphne_bench"
OUT_DIR       = "/tmp/benchmark/daphneDSL"

SIZES         = [1024, 2048, 4096]
SPARSITIES    = [0.001, 0.01, 0.05, 0.1, 0.2, 0.25, 1.0]
SEEDS         = list(range(10))

ANALYSES = {
    "min": ["mn = aggMin(R);"],
    "max": ["mx = aggMax(R);"],
    "mean":    ["mnv = mean(R);"],
    "symmetry": ["sym = sum(R == t(R)) == ($n * $n);"],   
}
type DT = Literal["double"] | Literal["int64_t"]
from typing import Generator, Final
i64: Final = "int64_t"
f64: Final = "double"

from abc import abstractmethod
class Spec:
    # Return generated matrices and their data type
    @staticmethod
    def rand() -> Generator[tuple[str, DT],None,None]:
        yield "X = rand($n, $n, -100.0, 100.0, $sp, $seed);", f64
        yield "X = rand($n, $n, -100, 100, $sp, $seed);", i64
    
    # Return generated code and the operation supplied as arg.
    @staticmethod
    @abstractmethod
    def gen() -> Generator[tuple[str, str],None,None]:
        pass
    
class EwUnary(Spec):
    @staticmethod
    def gen():
        # There are two cases. Add and Mul
        for op in ["exp", "-"]:
            yield f"R = {op}(X);", ("Minus" if op == "-" else op)        
    
class EwBinary(Spec):
    @staticmethod
    def rand():
        s1 =  "X = rand($n, $n, -100.0, 100.0, $sp, $seed);"
        # multiply seed by two to avoid same matrix twice
        s2 =  "Y = rand($n, $n, -100.0, 100.0, $sp, $seed * 2);"
        yield "\n".join([s1,s2]), f64
        yield "\n".join([s1,s2]).replace(".0",""), i64
    @staticmethod
    def gen():
        # There are two cases. Add and Mul
        for op in "+*":
            yield f"R = X {op} Y;", ("Add" if op == "+" else "Mul")        
class AggCol(Spec):
    @staticmethod
    def gen():
        for op in ["idxMin", "idxMax", "sum", "var", "mean", "stddev"]:
            yield f"R = {op}(X,1);\nR=as.f64(R);", op
class AggRow(Spec):
    @staticmethod
    def gen():
        for op in ["idxMin", "idxMax", "sum", "var", "mean", "stddev"]:
            yield f"R = {op}(X,0);\nR=as.f64(R);", op
class CTable(Spec):
    @staticmethod
    def rand():
        x = "X = rand($n, 1, 0, 100, $sp, $seed);"
        # multiply seed by two to avoid same matrix twice
        y = "Y = rand($n, 1, 0, 100, $sp, $seed * 2);"
        yield "\n".join([x,y]), i64
    @staticmethod
    def gen():
        yield "R1 = ctable(X,Y);\nR = as.f64(R1);", ""

def build_script_text(rand_gen: tuple[str, DT], gen: tuple[str, str], analyses):
    rand_code, dt = rand_gen
    gen_code, op = gen
    body      = [code_line for analysis in analyses for code_line in ANALYSES[analysis]]
    # get all variables used for analysis.
    varnames  = [ln.split("=")[0].strip() for ln in body]
    # for runs without any analysis, just add the sum.
    if not varnames: varnames.append("sum(as.f64(R))")
    # use all of them to ensure it is not compiled away.
    sink      = " + ".join(f"as.f64({v})" for v in varnames) if analyses else "sum(as.f64(R))"
    return f"""// {op=} {dt=} analysis={analyses}
{rand_code}
t0 = now();
{gen_code}
{"\n".join(body)}
t1 = now();
sink = {sink};
print(sink);
print(t1 - t0);
"""

def run_script(path, n, sp, seed, vectorized, fallback=False) -> tuple[int|None, bool]:
    args = [DAPHNE]
    if vectorized:
        args.append("--vec")
    args.append("--num-threads=1")
    if not fallback: args.append("--select-matrix-repr")
    args += [path, f"{n=}", f"{sp=}", f"{seed=}"]
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        last = proc.stderr if proc.stderr.strip() else "(no stderr)"
        if fallback: 
            print(f"  ERROR {fallback=} {os.path.basename(path)} n={n} sp={sp} vec={vectorized}: {last}")
            exit("i cant do it..")
        if not fallback: return run_script(path, n, sp, seed, vectorized, fallback=True)
        else: return None, fallback
    nums = re.findall(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?", proc.stdout)
    if not nums:
        print(f"  no numeric output: {proc.stdout}")
        return None, fallback
    return int(nums[-1]), fallback   # last printed number is the elapsed ns

def main():
    os.makedirs(SCRIPT_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    specs: list[type[Spec]] = [EwUnary, EwBinary, AggCol, AggRow, CTable]

    for spec in specs:
        out_csv = f"{OUT_DIR}/{spec.__name__}.csv"
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["mode", "op", "element_type", "analysis", "n", "sparsity", "seed", "time_ns", "select-matrix-repr"])
            # analyses = [()]
            analyses = [(),("symmetry",), ("mean",), ("min",), ("max",), ("min","max","mean","symmetry"), ("min","max")]
            for rand, gen, analysis in itertools.product(spec.rand(), spec.gen(), analyses):
                _, op = gen
                _, elem = rand
                print(f"{op=}, {elem=}, {analysis=}")
                if "symmetry" in analysis and spec in [CTable, AggCol, AggRow]:
                    analysis = list(analysis)
                    analysis.remove("symmetry") # type: ignore
                spath = os.path.join(SCRIPT_DIR, f"{spec.__name__}_{op}_{elem}_{"_".join(analysis)}.daphne")
                with open(spath, "w") as sf:
                    sf.write(build_script_text(rand, gen, analysis))
                print(f"Wrote script at {spath}")

                for n, sp, seed in itertools.product(SIZES, SPARSITIES, SEEDS):
                    for mode, vec in [("vectorized", True), ("naive", False)]:
                        ns, fallback = run_script(spath, n, sp, seed, vectorized=vec)
                        if ns is not None:
                            w.writerow([mode, op, elem, " ".join(analysis), n, sp, seed, ns, not fallback])
                            f.flush()
            print(f"done -> {out_csv}")
if __name__ == "__main__":
    main()
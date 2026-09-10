#!/usr/bin/env python3
"""Compare Q4_0 GPU perplexity between DP4 and DP8 activation builds."""

import argparse
import re
import subprocess
import sys
from pathlib import Path


def run_perplexity(executable: Path, model: Path, corpus: Path, device: str, context: int, batch: int, chunks: int, extra: list[str] | None = None) -> str:
    command = [
        str(executable),
        "-m", str(model),
        "-ngl", "99",
        "-dev", device,
        "-f", str(corpus),
        "-c", str(context),
        "-b", str(batch),
        "--chunks", str(chunks),
    ]
    if extra:
        command.extend(extra)
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
        output = result.stdout + result.stderr
        raise RuntimeError(
            f"{executable} exited with status {result.returncode}\n"
            f"command: {' '.join(command)}\n"
            f"{output}"
        )
    return result.stdout + result.stderr


def parse_ppl(output: str) -> float:
    matches = re.findall(r"Final estimate: PPL = ([0-9.eE+-]+)", output)
    if not matches:
        raise RuntimeError("could not find final PPL in perplexity output")
    return float(matches[-1])


def parse_metric(output: str, label: str) -> float:
    matches = re.findall(label + r"\s*(?:=|:)\s*([0-9.eE+-]+)", output)
    if not matches:
        raise RuntimeError(f"could not find {label} in perplexity output")
    return float(matches[-1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dp4-build", type=Path, required=True, help="build directory using the original DP4/Q8 activation path")
    parser.add_argument("--dp8-build", type=Path, required=True, help="build directory using GGML_CUDA_Q4_0_INT4_ACTIVATIONS")
    parser.add_argument("--dp8-reference-build", type=Path, help="build directory using GGML_CUDA_Q4_0_INT4_SCALAR_REFERENCE")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--device", default="ROCM2")
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--chunks", type=int, default=8)
    parser.add_argument("--max-relative-delta", type=float, help="fail if abs(PPL_DP8-PPL_DP4)/PPL_DP4 exceeds this value")
    args = parser.parse_args()

    for path, label in ((args.model, "model"), (args.corpus, "corpus")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}\nReplace the example placeholder with the actual path.")

    dp4 = args.dp4_build / "bin" / "llama-perplexity"
    dp8 = args.dp8_build / "bin" / "llama-perplexity"
    dp8_reference = args.dp8_reference_build / "bin" / "llama-perplexity" if args.dp8_reference_build else None
    for executable in (dp4, dp8, dp8_reference):
        if executable is None:
            continue
        if not executable.is_file():
            raise FileNotFoundError(f"missing executable: {executable}")

    ppl_dp4 = parse_ppl(run_perplexity(dp4, args.model, args.corpus, args.device, args.context, args.batch, args.chunks))
    ppl_dp8_output = run_perplexity(dp8, args.model, args.corpus, args.device, args.context, args.batch, args.chunks)
    ppl_dp8 = parse_ppl(ppl_dp8_output)
    absolute_delta = ppl_dp8 - ppl_dp4
    relative_delta = absolute_delta / ppl_dp4

    print(f"DP4 PPL: {ppl_dp4:.6f}")
    print(f"DP8 PPL: {ppl_dp8:.6f}")
    print(f"absolute delta: {absolute_delta:+.6f}")
    print(f"relative delta: {relative_delta:+.2%}")

    if dp8_reference is not None:
        reference_logits = Path("/tmp/q40-dot8-scalar-reference.logits")
        reference_output = run_perplexity(
            dp8_reference, args.model, args.corpus, args.device, args.context, args.batch, args.chunks,
            ["--save-all-logits", str(reference_logits)],
        )
        hardware_output = run_perplexity(
            dp8, args.model, args.corpus, args.device, args.context, args.batch, args.chunks,
            ["--kl-divergence", "--kl-divergence-base", str(reference_logits)],
        )
        print(f"DP8 scalar-reference PPL: {parse_ppl(reference_output):.6f}")
        print(f"DP8 hardware-vs-scalar KL: {parse_metric(hardware_output, r'Mean    KLD'):.6g}")
        print(f"DP8 hardware-vs-scalar probability RMS (%): {parse_metric(hardware_output, r'RMS [^:]+'):.6g}")
        print(f"DP8 hardware-vs-scalar same-top (%): {parse_metric(hardware_output, r'Same top p'):.6g}")

    if args.max_relative_delta is not None and abs(relative_delta) > args.max_relative_delta:
        print(
            f"FAIL: relative PPL delta {abs(relative_delta):.2%} exceeds "
            f"the limit {args.max_relative_delta:.2%}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

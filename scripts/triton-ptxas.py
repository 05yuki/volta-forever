#!/usr/bin/env python3
"""Run Triton's assembler and keep its PTX alongside the generated machine code."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


def main(arguments):
    compiler = shutil.which(os.environ.get("VF_TRITON_PTXAS", ""))
    if not compiler:
        sys.exit("vf: VF_TRITON_PTXAS does not name an executable ptxas")
    if Path(compiler).samefile(__file__):
        sys.exit("vf: VF_TRITON_PTXAS must name the NVIDIA assembler")
    if arguments == ["--version"]:
        result = subprocess.run([compiler, *arguments])
        if result.returncode == 0:
            # Triton hashes the complete version output into its compilation cache key.
            print("volta-forever PTX and cubin v2")
        return result.returncode

    # Triton's NVIDIA backend assembles one temporary PTX into an explicitly named
    # output. Keep other invocations, including tool discovery, with the assembler.
    sources = [Path(arg) for arg in arguments if arg.endswith(".ptx")]
    prefix = "--gpu-name=sm_"
    arch = next((arg[len(prefix):] for arg in arguments if arg.startswith(prefix)), "")
    if len(sources) != 1 or not re.fullmatch(r"[0-9]+[af]?", arch) or "-o" not in arguments:
        os.execv(compiler, [compiler, *arguments])
    out_index = arguments.index("-o") + 1
    if out_index == len(arguments):
        os.execv(compiler, [compiler, *arguments])
    destination = Path(arguments[out_index])
    tool_dirs = [Path(compiler).parent]
    if cuda_home := os.environ.get("CUDA_HOME"):
        tool_dirs.append(Path(cuda_home) / "bin")
    search_path = os.pathsep.join([*(str(path) for path in tool_dirs), os.environ.get("PATH", "")])
    fatbinary = shutil.which("fatbinary", path=search_path)

    # Short relative names keep commas in caller paths out of fatbinary's image syntax.
    with tempfile.TemporaryDirectory(prefix=".vf-triton-", dir=destination.parent) as tmp:
        work = Path(tmp).resolve()
        args = list(arguments)
        args[out_index] = str(work / "kernel.cubin")
        status = subprocess.run([compiler, *args]).returncode
        if status != 0:
            return status
        if not fatbinary:
            sys.exit("vf: install the CUDA toolkit or set CUDA_HOME so its fatbinary is available")
        # NVIDIA fatbinary can discard its first input line, even when this is
        # the PTX .version directive. Feed only the packer a disposable comment.
        with (work / "kernel.ptx").open("wb") as packed, sources[0].open("rb") as source:
            packed.write(b"// PTX source follows\n")
            shutil.copyfileobj(source, packed)
        status = subprocess.run([
            fatbinary, "--64", "--create=kernel.fatbin",
            f"--image3=kind=elf,sm={arch},file=kernel.cubin",
            f"--image3=kind=ptx,sm={arch},file=kernel.ptx",
        ], cwd=work).returncode
        if status != 0:
            return status
        os.replace(work / "kernel.fatbin", destination)
    return 0


if __name__ == "__main__":
    try:
        status = main(sys.argv[1:])
    except OSError as error:
        sys.exit(f"vf: could not assemble Triton kernel: {error}")
    sys.exit(128 - status if status < 0 else status)

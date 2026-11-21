"""Exercise the Triton assembler interface on hosts without a CUDA toolkit."""
import json
import os
from pathlib import Path
import subprocess
import shutil
import tempfile
import unittest


HELPER = Path(__file__).resolve().parents[2] / "scripts/triton-ptxas.py"


class AssemblerTest(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory(prefix="vf assembler,")
        self.addCleanup(scratch.cleanup)
        self.work = Path(scratch.name)
        self.ptx = self.work / "original source,.ptx"
        self.ptx.write_bytes(b"original PTX\n")
        self.out = self.work / "Triton output,.o"
        self.out.write_bytes(b"old output")
        self.argv_file = self.work / "argv.json"
        self.env = dict(os.environ, VF_TRITON_PTXAS=str(self.work / "ptxas"),
                        TEST_ARGV=str(self.argv_file))
        self.tool("ptxas", """import json, os, pathlib, sys
args = sys.argv[1:]
pathlib.Path(os.environ['TEST_ARGV']).write_text(json.dumps(args))
if '--version' in args:
    print('NVIDIA ptxas release 12.9')
if os.environ.get('TEST_PTXAS_FAIL'):
    sys.exit(31)
if '-o' in args:
    pathlib.Path(args[args.index('-o') + 1]).write_bytes(b'original cubin')
""")
        self.tool("fatbinary", """import os, pathlib, sys
assert sys.argv[1:] == ['--64', '--create=kernel.fatbin',
    '--image3=kind=elf,sm=80,file=kernel.cubin',
    '--image3=kind=ptx,sm=80,file=kernel.ptx']
if os.environ.get('TEST_PACK_FAIL'):
    sys.exit(32)
out = pathlib.Path('kernel.cubin').read_bytes() + b'|' + pathlib.Path('kernel.ptx').read_bytes()
pathlib.Path('kernel.fatbin').write_bytes(out)
""")
        self.args = ["-lineinfo", "--gpu-name=sm_80", str(self.ptx), "-o", str(self.out)]

    def tool(self, name, source):
        path = self.work / name
        path.write_text("#!/usr/bin/env python3\n" + source)
        path.chmod(0o755)

    def invoke(self, *args, **env):
        return subprocess.run([str(HELPER), *args], env={**self.env, **env},
                              capture_output=True, text=True)

    def test_compile(self):
        result = self.invoke(*self.args)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.out.read_bytes(), b"original cubin|// PTX source follows\noriginal PTX\n")
        self.assertEqual(self.ptx.read_bytes(), b"original PTX\n")
        actual = json.loads(self.argv_file.read_text())
        self.assertEqual(actual[:-1], self.args[:-1])

    def test_version_and_passthrough(self):
        version = self.invoke("--version")
        self.assertEqual(version.returncode, 0)
        self.assertIn("NVIDIA ptxas release 12.9", version.stdout)
        self.assertIn("volta-forever", version.stdout)
        result = self.invoke("--help", "argument with spaces", "")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(json.loads(self.argv_file.read_text()), ["--help", "argument with spaces", ""])

    def test_failures(self):
        for env, status in [("TEST_PTXAS_FAIL", 31), ("TEST_PACK_FAIL", 32)]:
            with self.subTest(env=env):
                result = self.invoke(*self.args, **{env: "1"})
                self.assertEqual(result.returncode, status, result.stderr)
                self.assertEqual(self.out.read_bytes(), b"old output")
                self.assertFalse(list(self.work.glob(".vf-triton-*")))
        result = self.invoke("--version", TEST_PTXAS_FAIL="1")
        self.assertEqual(result.returncode, 31)
        self.assertNotIn("volta-forever", result.stdout)


class NvidiaToolsTest(unittest.TestCase):
    def test_first_directive_survives_packing(self):
        cuda_bin = str(Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "bin")
        path = os.pathsep.join([cuda_bin, os.environ.get("PATH", "")])
        compiler = shutil.which("ptxas", path=path)
        dump = shutil.which("cuobjdump", path=path)
        if not compiler or not dump or not shutil.which("fatbinary", path=path):
            self.skipTest("NVIDIA toolkit unavailable")
        with tempfile.TemporaryDirectory() as scratch:
            work = Path(scratch)
            ptx = work / "input.ptx"
            env = dict(os.environ, VF_TRITON_PTXAS=compiler)
            directives = ".version 8.0\n.target sm_80\n.address_size 64\n"
            for prefix in ("", "// user comment\n"):
                with self.subTest(prefix=prefix):
                    source = prefix + directives + ".visible .entry test() { ret; }\n"
                    ptx.write_text(source)
                    subprocess.run([HELPER, "--gpu-name=sm_80", ptx, "-o", "test.fatbin"],
                                   env=env, cwd=work, check=True)
                    subprocess.run([dump, "-xptx", "all", "test.fatbin"], cwd=work,
                                   check=True, capture_output=True)
                    extracted = next(work.glob("*.sm_80.ptx"))
                    self.assertIn(directives, extracted.read_text())
                    self.assertEqual(ptx.read_text(), source)
                    extracted.unlink()


if __name__ == "__main__":
    unittest.main()

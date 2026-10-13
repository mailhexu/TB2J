"""No-JAX default-path tests (story 010-3).

Asserts the default CPU import/execution path does not require JAX: importing
``TB2J`` and the default CPU interfaces must not pull in ``TB2J.gpu`` or ``jax``.
Each check runs in an isolated subprocess where JAX is blocked from the start,
so the result is deterministic and independent of test ordering.

JAX optionality is already implemented via lazy ``jax_utils`` proxies; this test
guards against regressions where a module-level ``from TB2J.gpu... import`` sneaks
onto the default import path (as ``interfaces/manager.py`` and
``interfaces/siesta_interface.py`` did before this story).
"""

from __future__ import annotations

import subprocess
import sys
import textwrap


def _run_isolated(script: str) -> subprocess.CompletedProcess:
    """Run a snippet in a clean subprocess; return the CompletedProcess."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_default_imports_do_not_pull_gpu_or_jax():
    """Importing TB2J + default CPU interfaces must not import TB2J.gpu or jax."""
    result = _run_isolated(
        """
        import sys
        # Block jax so any accidental eager `import jax` fails loudly. (A None
        # entry makes `import jax` raise ImportError.)
        sys.modules["jax"] = None
        # Sanity check: jax is genuinely unimportable now.
        try:
            import jax  # noqa: F401
        except ImportError:
            pass
        else:
            raise SystemExit("jax was not blocked")
        # If any default module eagerly imports jax, the next line raises
        # ImportError. Successful import proves no eager jax dependency.
        import TB2J
        import TB2J.exchangeCL2
        import TB2J.interfaces.wannier90_interface
        import TB2J.interfaces.manager
        import TB2J.interfaces.abacus.gen_exchange_abacus
        import TB2J.interfaces.siesta_interface
        gpu = sorted(m for m in sys.modules if m.startswith("TB2J.gpu"))
        assert not gpu, f"default import pulled in TB2J.gpu: {gpu}"
        print("OK")
        """
    )
    assert (
        result.returncode == 0
    ), f"default import pulled JAX/GPU:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    assert result.stdout.strip() == "OK"


def test_gpu_module_imports_without_jax_but_requires_it_at_use():
    """TB2J.gpu imports cleanly without JAX; kernels raise only when used."""
    result = _run_isolated(
        """
        import sys
        sys.modules["jax"] = None
        # Importing the gpu package must NOT require jax (lazy proxies).
        import TB2J.gpu  # noqa: F401
        from TB2J.gpu.jax_utils import _check_jax, _require_jax
        assert _check_jax() is False, "_check_jax should be False with jax blocked"
        try:
            _require_jax()
        except ImportError:
            pass
        else:
            raise AssertionError("_require_jax should raise ImportError without jax")
        print("OK")
        """
    )
    assert (
        result.returncode == 0
    ), f"gpu lazy-import contract broken:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    assert result.stdout.strip() == "OK"


def test_default_exchange_runs_without_jax(tmp_path):
    """A real CPU Wannier exchange runs with JAX blocked and yields a loadable SpinIO.

    Covers story 010-3 AC2: not just imports, but an actual CPU calculation must run
    without JAX and produce a loadable canonical result. Uses the smallest bundled
    Wannier input (SrMnO3) at a reduced k-mesh/nz so it stays fast; skips when the
    governed input is absent.
    """
    from conftest import require_input

    input_dir = require_input(
        "inputs/2_SrMnO3_wannier/data", "Wannier90 collinear", "SrMnO3"
    )
    out_dir = tmp_path / "TB2J_results"
    script = f"""
        import sys
        sys.modules["jax"] = None  # block JAX; any eager `import jax` fails loudly
        sys.argv = [
            "wann2J.py",
            "--path", {str(input_dir)!r},
            "--posfile", "abinit.in",
            "--efermi", "6.15",
            "--kmesh", "2", "2", "2",
            "--nz", "10",
            "--elements", "Mn",
            "--prefix_up", "abinito_w90_up",
            "--prefix_down", "abinito_w90_down",
            "--output_path", {str(out_dir)!r},
        ]
        from TB2J.scripts.wann2J import run_wann2J

        run_wann2J()
        print("OK")
    """
    result = _run_isolated(script)
    assert (
        result.returncode == 0
    ), f"CPU exchange failed with JAX blocked:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    # The canonical result must be loadable as a SpinIO.
    pickle_path = out_dir / "TB2J.pickle"
    assert pickle_path.exists(), f"TB2J.pickle not written:\n{result.stdout}"
    from TB2J.io_exchange.io_exchange import SpinIO

    SpinIO.load_pickle(str(out_dir))  # loads without error

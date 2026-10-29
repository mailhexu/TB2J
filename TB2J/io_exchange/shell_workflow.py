"""File-level workflows for editable symmetry-shell exchange documents."""

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

from ase.data import atomic_numbers
from ase.io import read

from . import shells


def _generate_native(
    source,
    output,
    *,
    mode,
    symmetry,
    symprec,
    magnetic_tol,
    project,
    equivalent_comments,
):
    from .shell_context import write_context

    companion = output.with_suffix(".context.npz")
    paths = (source, output, companion)
    for i, first in enumerate(paths):
        for second in paths[i + 1 :]:
            if first.resolve() == second.resolve() or (
                first.exists() and second.exists() and os.path.samefile(first, second)
            ):
                raise ValueError(
                    "input, shell output, and context must be distinct files"
                )
    from .io_exchange import SpinIO

    spinio = SpinIO.load_pickle(path=source.parent, fname=source.name)
    spinio._shell_source_format = "pickle"
    spinio._shell_precision_profile = "exact"
    anisotropic = any(
        getattr(spinio, name, None) is not None for name in ("dmi_ddict", "Jani_dict")
    )
    if mode is None:
        mode = "tensor" if anisotropic else "isotropic"
    elif mode == "isotropic" and anisotropic:
        raise ValueError("mode isotropic cannot discard present DMI or Jani channels")
    result = shells.compress(
        spinio,
        symmetry=symmetry,
        mode=mode,
        symprec=symprec,
        magnetic_tol=magnetic_tol,
        on_inconsistent="project" if project else "error",
    )
    stages = []
    try:
        for suffix in (".npz", ".toml"):
            with NamedTemporaryFile(
                dir=output.parent, prefix=".shell-", suffix=suffix, delete=False
            ) as stream:
                stages.append(Path(stream.name))
        digest = write_context(spinio, stages[0])
        document = replace(result.document, context_sha256=digest)
        shells.write_shells(
            document, stages[1], equivalent_comments=equivalent_comments
        )
        os.replace(stages[0], companion)
        os.replace(stages[1], output)
    finally:
        for stage in stages:
            stage.unlink(missing_ok=True)
    return shells.ShellResult(document=document, report=result.report)


def generate_shells(
    source,
    output="exchange.shells.toml",
    *,
    magnetic_species=None,
    magnetic_atoms=None,
    cutoff=None,
    mode=None,
    symmetry="sg",
    reference_spins=None,
    symprec=1e-5,
    magnetic_tol=1e-5,
    project=False,
    equivalent_comments=True,
):
    """Generate an annotated shell document from a structure or trusted pickle.

    Magnetic atom indices are one-based over the complete structure; species
    select all atoms with the given chemical symbols. The cutoff is in Angstrom.
    Pickle input must be trusted because loading it can execute arbitrary code.
    """
    if not isinstance(project, bool):
        raise ValueError("project must be a boolean")
    source = Path(source)
    output = Path(output)
    if source.suffix.lower() in {".pickle", ".pkl"}:
        if any(
            value is not None
            for value in (magnetic_species, magnetic_atoms, cutoff, reference_spins)
        ):
            raise ValueError(
                "pickle input does not accept structure selectors, cutoff, or reference spins"
            )
        return _generate_native(
            source,
            output,
            mode=mode,
            symmetry=symmetry,
            symprec=symprec,
            magnetic_tol=magnetic_tol,
            project=project,
            equivalent_comments=equivalent_comments,
        )
    if (magnetic_species is None) == (magnetic_atoms is None):
        raise ValueError(
            "structure input requires exactly one of magnetic_species or magnetic_atoms"
        )
    if project:
        raise ValueError("project applies only to imported exchange values")
    if source.resolve() == output.resolve() or (
        output.exists() and os.path.samefile(source, output)
    ):
        raise ValueError("output must not overwrite the input structure")
    atoms = read(source)
    if magnetic_species is not None:
        if isinstance(magnetic_species, str):
            magnetic_species = [magnetic_species]
        species = tuple(magnetic_species)
        if not species or any(
            not isinstance(s, str) or s not in atomic_numbers for s in species
        ):
            raise ValueError("magnetic_species must contain valid chemical symbols")
        symbols = atoms.get_chemical_symbols()
        missing = sorted(set(species) - set(symbols))
        if missing:
            raise ValueError(
                "magnetic_species not present in structure: " + ", ".join(missing)
            )
        selected = set(species)
        magnetic_atoms = [
            i for i, symbol in enumerate(symbols, 1) if symbol in selected
        ]
    document = shells.template(
        atoms,
        magnetic_atoms,
        cutoff,
        symmetry=symmetry,
        reference_spins=reference_spins,
        mode="isotropic" if mode is None else mode,
        symprec=symprec,
        magnetic_tol=magnetic_tol,
    )
    report = shells.ConversionReport()
    # Use sibling staging so validation or serialization failures cannot truncate
    # the user's existing output. Preserve the requested path until publication.
    temporary = None
    try:
        with NamedTemporaryFile(
            dir=output.parent, prefix=".shell-", suffix=".toml", delete=False
        ) as stream:
            temporary = Path(stream.name)
        shells.write_shells(
            document, temporary, equivalent_comments=equivalent_comments
        )
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return shells.ShellResult(document=document, report=report)


_OUTPUT_MANIFEST = ".tb2j-shell-outputs.json"


def _file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_leaf(root, relative):
    """Refuse links and non-directories in a path to an owned output."""
    path = root
    for part in Path(relative).parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"output path is a symlink: {path}; choose a fresh output")
        if path != root / relative and path.exists() and not path.is_dir():
            raise ValueError(f"output parent is not a directory: {path}")
    return path


def _publish_results(stage, output, plan, protected):
    """Publish only files with verified ownership; preserve unrelated entries."""
    from .io_exchange import _RESULT_TARGET_FILES

    candidates = set(plan.written_paths) | set(plan.omitted_paths)
    candidates.update(
        path for target in _RESULT_TARGET_FILES.values() for path in target
    )
    candidates.update(
        (
            "TB2J.pickle",
            "structure.vasp",
            "exchange.out",
            "JvsR.pdf",
            "exchange_orb_decomposition.out",
        )
    )
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise ValueError("output must be a real directory; choose a fresh output")
    manifest_path = _safe_leaf(output, _OUTPUT_MANIFEST)
    owned = {}
    if manifest_path.exists():
        if not manifest_path.is_file():
            raise ValueError("invalid output manifest; choose a fresh output")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                not isinstance(manifest, dict)
                or set(manifest) != {"version", "files"}
                or manifest["version"] != 1
                or not isinstance(manifest["files"], dict)
            ):
                raise ValueError("invalid manifest")
            owned = manifest["files"]
            if any(
                name not in candidates
                or not isinstance(value, str)
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
                for name, value in owned.items()
            ):
                raise ValueError("invalid manifest entries")
        except (ValueError, OSError, TypeError) as exc:
            raise ValueError("invalid output manifest; choose a fresh output") from exc
    # Validate all candidate destinations and input aliases before any publication.
    for name in candidates | {_OUTPUT_MANIFEST}:
        path = _safe_leaf(output, name)
        if not path.exists():
            continue
        if not path.is_file():
            raise ValueError(f"output collision at {path}; choose a fresh output")
        if any(p.exists() and os.path.samefile(path, p) for p in protected):
            raise ValueError(f"output aliases an input: {path}; choose a fresh output")
        if name != _OUTPUT_MANIFEST and (
            name not in owned or _file_digest(path) != owned[name]
        ):
            raise ValueError(
                f"ambiguous output ownership at {path}; choose a fresh output"
            )
    new_owned = {name: _file_digest(stage / name) for name in plan.written_paths}
    (stage / _OUTPUT_MANIFEST).write_text(
        json.dumps({"version": 1, "files": new_owned}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output.mkdir(exist_ok=True)
    for name in plan.written_paths:
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage / name, destination)
    for name in owned.keys() - new_owned.keys():
        path = output / name
        if path.exists():
            path.unlink()
    os.replace(stage / _OUTPUT_MANIFEST, manifest_path)


def export_shells(source, output="TB2J_results"):
    """Validate and expand shell TOML into standard TB2J results."""
    from .shell_context import load_context

    source = Path(source)
    output = Path(output)
    document = shells.read_shells(source)
    shells.validate_shells(document, for_export=True)
    base = (
        load_context(source, document)
        if document.source_format == "pickle" or document.context_sha256 is not None
        else None
    )
    result = shells.expand(document, base=base)
    for field in ("NJT_Jdict", "NJT_ddict"):
        for key, vector in (getattr(result.spinio, field, None) or {}).items():
            shells._numbers(vector, (3,), f"{field} {key!r}")
    result.spinio.results_plan()
    companion = source.with_suffix(".context.npz")
    protected = [source]
    if document.source_format == "pickle":
        protected.append(companion)
    with TemporaryDirectory(prefix=".tb2j-shell-", dir=output.parent) as temporary:
        stage = Path(temporary)
        plan = result.spinio.write_all(stage)
        _publish_results(stage, output, plan, protected)
    return replace(result, results=plan)

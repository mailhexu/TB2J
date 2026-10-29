"""File-level workflows for editable symmetry-shell exchange documents."""

import os
from dataclasses import replace
from pathlib import Path
from tempfile import NamedTemporaryFile

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
    spinio = shells.read_exchange(source, "pickle").spinio
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

"""File-level workflows for editable symmetry-shell exchange documents."""

import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from ase.data import atomic_numbers
from ase.io import read

from . import shells


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
    """Generate an annotated, unfilled shell document from a structure file.

    Magnetic atom indices are one-based over the complete structure; species
    select all atoms with the given chemical symbols. The cutoff is in Angstrom.
    """
    if (magnetic_species is None) == (magnetic_atoms is None):
        raise ValueError(
            "structure input requires exactly one of magnetic_species or magnetic_atoms"
        )
    if project:
        raise ValueError("project applies only to imported exchange values")
    source = Path(source)
    output = Path(output)
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

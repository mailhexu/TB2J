"""Generation-time symmetry witnesses for compact exchange-shell comments."""

import numpy as np

from TB2J import symmetry_core


def _pair_label(pair, labels):
    return " ".join(
        (labels[pair[0] - 1], labels[pair[1] - 1], *(str(x) for x in pair[2:]))
    )


def pair_comment_sections(
    document, shell, orbit, representative, *, validated=None
) -> str:
    """Render aligned witnesses from a stored representative to each physical pair.

    Operation indices deliberately follow ``document.operations`` without
    reordering: those are also the operations serialized in the TOML document.
    """
    from .shells import _core_key, _physical_pairs, _reverse_core, _validated_operations

    cell = np.asarray(document.cell_angstrom, dtype=float)
    positions = np.asarray(document.positions_frac, dtype=float)
    operations, site_maps = (
        validated
        if validated is not None
        else _validated_operations(document, cell, positions)
    )
    representative_key = _core_key(representative)
    image_maps, _ = symmetry_core.pair_images(
        (representative_key,), operations, site_maps, positions, cell.T
    )
    if len(image_maps) != len(document.operations):
        raise ValueError(
            "symmetry.operations: pair-image order does not match the stored snapshot"
        )

    def witness(target):
        target_key = _core_key(target)
        direct = None
        reversed_bond = None
        for index, image_map in enumerate(image_maps, 1):
            image = image_map[representative_key]
            if image == target_key and direct is None:
                direct = index
            elif _reverse_core(image) == target_key and reversed_bond is None:
                reversed_bond = index
        if direct is not None:
            return direct, "false"
        if reversed_bond is not None:
            return reversed_bond, "true"
        raise ValueError(
            f"shell {shell.id}: no stored operation maps representative to pair {target}"
        )

    pairs = _physical_pairs(orbit)
    headings = ("atom1", "atom2", "Rx", "Ry", "Rz", "operation", "reversed")
    rows = []
    for pair in pairs:
        operation, reversed_bond = witness(pair)
        rows.append(
            (
                document.site_labels[pair[0] - 1],
                document.site_labels[pair[1] - 1],
                *(str(value) for value in pair[2:]),
                str(operation),
                reversed_bond,
            )
        )
    widths = [max(len(value) for value in column) for column in zip(headings, *rows)]

    lines = [
        f"# Generated pair witnesses from representative "
        f"{_pair_label(representative, document.site_labels)}.\n",
        "# Editing the representative makes these comments stale.\n",
    ]
    lines.extend(
        "# "
        + "  ".join(f"{value:>{width}}" for value, width in zip(row, widths))
        + "\n"
        for row in (headings, *rows)
    )
    return "".join(lines)

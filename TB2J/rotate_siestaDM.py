import sisl


def rotate_siesta_DM(DM, noncollinear=False):
    angles_list = [[0.0, 90.0, 0.0], [0.0, 90.0, 90.0]]
    if noncollinear:
        angles_list += [[0.0, 45.0, 0.0], [0.0, 90.0, 45.0], [0.0, 45.0, 90.0]]

    for angles in angles_list:
        yield DM.spin_rotate(angles)


def read_label(fdf_fname):
    label = "siesta"
    with open(fdf_fname, "r") as File:
        for line in File:
            corrected_line = line.lower().replace(".", "").replace("-", "")
            if "systemlabel" in corrected_line:
                label = line.split()[1]
                break

    return label


def rotate_DM(fdf_fname, noncollinear=False):
    fdf = sisl.get_sile(fdf_fname)
    DM = fdf.read_density_matrix()
    label = read_label(fdf_fname)

    rotated = rotate_siesta_DM(DM, noncollinear=noncollinear)

    for i, rotated_DM in enumerate(rotated):
        rotated_DM.write(f"{label}_{i+1}.DM")
    DM.write(f"{label}_0.DM")

    print(
        f"The output has been written to the {label}_i.DM files. {label}_0.DM contains the reference density matrix."
    )

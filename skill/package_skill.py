import argparse
import os
import zipfile


def package_skill(skill_dir, output_file=None):
    """Packages a skill directory into a .skill file (zip archive)."""

    skill_dir = os.path.abspath(skill_dir)
    if not os.path.isdir(skill_dir):
        print(f"Error: Directory '{skill_dir}' does not exist.")
        return

    if output_file is None:
        output_file = f"{os.path.basename(skill_dir)}.skill"

    print(f"Packaging '{skill_dir}' into '{output_file}'...")

    try:
        with zipfile.ZipFile(output_file, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(skill_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, skill_dir)
                    zipf.write(file_path, arcname)
                    print(f"  Added: {arcname}")

        print(f"Successfully created '{output_file}'.")

    except Exception as e:
        print(f"Error creating skill package: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Package a directory into a .skill file."
    )
    parser.add_argument("skill_dir", help="Directory containing the skill files")
    parser.add_argument("--output", "-o", help="Output .skill file path (optional)")

    args = parser.parse_args()
    package_skill(args.skill_dir, args.output)

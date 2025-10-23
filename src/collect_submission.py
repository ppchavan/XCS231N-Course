import zipfile, os


def make_submission_zip(output_filename="a3.zip", submission_dir="submission"):

    print(f":package: Creating {output_filename} from {submission_dir}/ ...")

    if os.path.exists(output_filename):
        os.remove(output_filename)
        
    with zipfile.ZipFile(output_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(submission_dir):
            dirs[:] = [
                d for d in dirs if d not in ("__pycache__", ".ipynb_checkpoints")
            ]
            for file in files:
                if file.endswith((".py", ".ipynb")):
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(
                        file_path, start=submission_dir
                    )  # :white_check_mark: correct relative path
                    zipf.write(file_path, arcname)
                    print("  +", arcname)
    print(f":white_check_mark: Done! Created {output_filename} successfully.")


# :point_down: Run it here
make_submission_zip()

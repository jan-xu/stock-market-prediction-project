from pathlib import Path

RUN_SUPER_FOLDER = Path("./runs")
if not RUN_SUPER_FOLDER.exists():
    RUN_SUPER_FOLDER.mkdir()

SUBFOLDERS = [
    "models",  # .pt model files
    "logs",  # console printout logs
    "data",  # misc data files
    "figures",  # figures
    "figures/png",  # png figures
    "figures/html",  # html figures
]


def create_run_folder(run_name):

    # Create run folder
    run_folder = RUN_SUPER_FOLDER / run_name
    run_folder.mkdir()
    print(f"Created run folder: {run_folder}\n")

    # Create subfolders
    for subfolder in SUBFOLDERS:
        subfolder_path = run_folder / subfolder
        subfolder_path.mkdir()

    return run_folder

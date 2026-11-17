"""
Shared setup for every notebook in this repository.

Each notebook's first cell imports this module and calls `setup()`. That does three things:

1. Picks one working folder, WORK, where every notebook reads its inputs and writes its outputs.
   Set the environment variable CIC_WORK to choose it yourself. Otherwise:
     - on Google Colab, a folder on your Google Drive (so a 10-hour run survives a disconnect);
       set CIC_DRIVE=0 to stay on Colab's local disk instead
     - anywhere else (Jupyter, VS Code, PyCharm), <repository>/work
2. Changes the current directory to WORK, so the notebooks' relative paths land there.
3. Prints where things are, so a reader can see what will be read and written.

`require(name)` returns the path of an input in WORK. If the file is not there yet, it is copied
from the repository (refactoring_levels/refactoring_levels.yaml) or downloaded from this study's
Zenodo record or Hugging Face model repository. That is what lets a reader start
at any stage without re-running the ones before it.
"""

import os
import shutil
import sys
import urllib.request
from pathlib import Path

# --------------------------------------------------------------------------------------------
# Where the published pieces live. Fill in the two TODOs once the Zenodo DOI is reserved and the
# Hugging Face repository exists; nothing else in the repository needs to change.
# --------------------------------------------------------------------------------------------
GITHUB_REPO = "https://github.com/Binamra00/commit-intent-and-refactoring-levels"
LEVIN_ZENODO_RECORD = "835534"     # Levin & Yehudai's labelled commits (their record)
CORPUS_ZENODO_RECORD = "24182007"  # the five-repository corpus mined for the thesis
STUDY_ZENODO_RECORD = "24295072"   # this study: doi.org/10.5281/zenodo.24295072
HF_MODEL_REPO = None               # TODO: e.g. "Binamra00/codebert-commit-intent"

ROOT = Path(__file__).resolve().parent
COMMITTED = [ROOT / "refactoring_levels"]

# What each published record holds. Every file is uploaded on its own, not zipped, so it can be
# downloaded by name.
ON_ZENODO = {
    # inputs and outputs too large for git
    "levin_with_diffs.jsonl", "stage5_logits.jsonl", "dataset.csv", "dataset_meta.json",
    # small stage outputs, so any notebook can start without re-running the ones before it
    "levin_columns.json", "stage2_train_idx.npy", "stage2_test_idx.npy", "stage2_summary.json",
    "stage3_summary.json", "stage3b_decision.json", "stage4a_decision.json",
    "stage4b_results.json", "stage4c_calibration.json", "stage5a_meta.json",
}
ON_HUGGING_FACE = {f"model_attention_{s}.pt" for s in (42, 1337, 2024)}

# Which notebook writes each file, for the error message when one is missing.
PRODUCED_BY = {
    "levin_dataset.pkl": "levin_classifier/01_levin_dataset.ipynb",
    "levin_columns.json": "levin_classifier/01_levin_dataset.ipynb",
    "stage2_train_idx.npy": "levin_classifier/02_reproduce_levin.ipynb",
    "stage2_test_idx.npy": "levin_classifier/02_reproduce_levin.ipynb",
    "levin_with_diffs.jsonl": "levin_classifier/03a_extract_diffs.ipynb",
    "stage4b_results.json": "levin_classifier/04b_finetune.ipynb",
    "stage5a_commits.jsonl": "levin_classifier/05a_assemble.ipynb",
    "refactoring_levels.yaml": "refactoring_levels/refactoring_taxonomy.ipynb",
    **{n: "levin_classifier/04b_finetune.ipynb" for n in ON_HUGGING_FACE},
}

WORK = None


def in_colab():
    return "google.colab" in sys.modules


def _work_dir():
    if os.environ.get("CIC_WORK"):
        return Path(os.environ["CIC_WORK"]).expanduser()
    if in_colab():
        if os.environ.get("CIC_DRIVE", "1") != "0":
            from google.colab import drive
            if not Path("/content/drive/MyDrive").exists():
                drive.mount("/content/drive")
            return Path("/content/drive/MyDrive") / f"{ROOT.name}-work"
        return Path("/content") / f"{ROOT.name}-work"
    return ROOT / "work"


def setup():
    """Choose WORK, move into it, and report. Safe to call more than once."""
    global WORK
    WORK = _work_dir().resolve()
    WORK.mkdir(parents=True, exist_ok=True)
    os.chdir(WORK)
    os.environ["CIC_ROOT"] = str(ROOT)          # lets a re-run of the setup cell find the repo
    print(f"repository : {ROOT}")
    print(f"WORK       : {WORK}   (inputs are read and outputs written here)")
    # torch is deliberately not imported here: 05b must set PYTORCH_CUDA_ALLOC_CONF before
    # anything initialises CUDA.
    return WORK


def _download(url, dest):
    print(f"   downloading {dest.name} ...", end=" ", flush=True)
    tmp = dest.with_name(dest.name + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
    print(f"done ({dest.stat().st_size / 1e6:.0f} MB)")


def require(name, optional=False):
    """
    Path to `name` in WORK, fetching it first if needed.

    Order: already in WORK -> refactoring_levels/ in the repository -> this study's Zenodo record ->
    Hugging Face. With optional=True a missing file returns None instead of stopping.
    """
    if WORK is None:
        raise RuntimeError("call setup() first")
    dest = WORK / name
    if dest.exists():
        return dest

    for d in COMMITTED:
        src = d / name
        if src.exists():
            shutil.copy2(src, dest)
            print(f"   {name}: copied from {src.relative_to(ROOT)}")
            return dest

    if name in ON_ZENODO and STUDY_ZENODO_RECORD:
        _download(f"https://zenodo.org/records/{STUDY_ZENODO_RECORD}/files/{name}?download=1",
                  dest)
        return dest

    if name in ON_HUGGING_FACE and HF_MODEL_REPO:
        _download(f"https://huggingface.co/{HF_MODEL_REPO}/resolve/main/{name}", dest)
        return dest

    if optional:
        return None
    hint = PRODUCED_BY.get(name)
    raise SystemExit(f"{name} is not in {WORK}"
                     + (f" -- run {hint} first" if hint else "")
                     + (" or set STUDY_ZENODO_RECORD / HF_MODEL_REPO in replication.py"
                        if name in ON_ZENODO | ON_HUGGING_FACE else ""))

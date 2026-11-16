# Commit Intent and Refactoring Levels

Two pieces of work on the same five Java repositories, published together because one feeds the
other.

1. **A commit-intent classifier.** CodeBERT, fine-tuned on Levin & Yehudai's 1,151 commits
   labelled corrective, perfective or adaptive (Swanson's maintenance categories), reading the
   commit message and the per-file diff. It is then calibrated and applied to 34,609 commits.
2. **Refactoring levels.** Every refactoring type RefactoringMiner reports, assigned Murphy-Hill,
   Parnin & Black's high, medium or low abstraction level, with the reason for each.

Neither is part of the author's M.S. thesis, but both use its corpus
([10.5281/zenodo.24182007](https://doi.org/10.5281/zenodo.24182007)).

## Results

### Commit-intent classifier

All figures except the frozen-encoder row are on the same 173 held-out commits that Stage 2's
reproduction of Levin & Yehudai is scored on; that row is on a validation split of the training
commits.

| model | accuracy | Cohen's κ |
|---|---|---|
| Levin & Yehudai, as published (RF, Keywords / Combined) | 76.7% | 0.635 |
| Our reproduction of that model | 75.7% | 0.621 |
| Best reproduced feature-based model (GBM, Combined / Combined) | 78.0% | 0.654 |
| Frozen CodeBERT + pooling head (validation) | 63.3% | 0.441 |
| **Fine-tuned CodeBERT, attention pooling, mean of 3 seeds** | **86.7% ± 1.2** | **0.791 ± 0.020** |
| Fine-tuned CodeBERT, transformer pooling, mean of 3 seeds | 84.4% ± 2.1 | 0.756 ± 0.032 |

**Calibration.** Temperature scaling (T = 2.86) took expected calibration error from 0.109 to
0.025. On the held-out commits, setting aside those with a calibrated margin below 0.34 keeps
93.1% of them at 90.7% precision.

**Applied to the thesis corpus.** 34,609 non-merge commits across commons-lang, checkstyle, dubbo,
junit-framework and questdb. At the primary margin threshold of 0.47, 21.4% are left
unclassified. On the 4,422 commits whose linked issue or PR labels imply exactly one class, the
classifier agrees with the maintainers' label 58.3% of the time overall, rising from 37.2% in the
lowest-confidence decile to 78.3% in the second-highest.

**Read these with their limits.**

- The test set is 173 commits. Seed 2024, which Stage 5 uses, was the best of three on those
  commits, so the per-seed figure for it (87.9%, κ 0.81) is optimistic; the mean is the number to
  quote.
- Levin & Yehudai dropped commits they could not classify confidently, so the labelled data
  contains no ambiguous commits. The calibration threshold is fitted on that population and is
  likely too permissive on an unfiltered corpus.
- Maintainer labels are a weak reference: many describe process (`approved`, `dependencies`)
  rather than the kind of work, and only unambiguous ones are used.
- scikit-learn has no C4.5, so the J48 reproduction uses CART and is approximate.

### Refactoring levels

| check | result |
|---|---|
| Types classified | 107 (61 high, 21 medium, 25 low) |
| The paper's own assignments reproduced | 18 / 18 |
| Agreement with an independent application (Horikawa et al., arXiv:2511.04824) | 93 / 102 shared types (91.2%) |
| Refactoring instances in the corpus | 278,923: 60.0% high, 18.8% medium, 21.2% low |
| Murphy-Hill et al.'s Eclipse distribution, for comparison | 60% high, 22% medium, 18% low |

Nine types are classified differently from Horikawa et al. Four follow from the paper's
Figure 2 (two directly, two by the same reasoning) and two from its definition of medium. The other three (Rename Parameter, Add and Remove
Parameter Modifier) turn on whether "signature" means the declaration line or the caller-visible
contract. Moving all nine to the other reading shifts the corpus-wide high share by 5.4 points,
and commons-lang's by 16.2, so level-dependent results should be reported both ways. The mapping,
with the reason for every type, is `refactoring_levels/refactoring_levels.yaml`.

## Repository layout

```
├── replication.py              shared setup: the working folder, and fetching missing inputs
├── requirements.txt
├── levin_classifier/
│   ├── 01_levin_dataset.ipynb          read and check Levin & Yehudai's dataset
│   ├── 02_reproduce_levin.ipynb        reproduce their classifiers; fix the train/test split
│   ├── 03a_extract_diffs.ipynb         fetch each labelled commit's diff from GitHub
│   ├── 03b_diff_representation.ipynb   how to feed a commit to a 512-token model
│   ├── 04a_architecture.ipynb          choose the pooling, encoder frozen
│   ├── 04b_finetune.ipynb              fine-tune end to end; the only notebook that trains
│   ├── 04c_calibration.ipynb           temperature scaling and the confidence threshold
│   ├── 05a_assemble.ipynb              build one record per corpus commit (no GPU)
│   └── 05b_classify.ipynb              classify, calibrate, check, write dataset_a.csv
├── refactoring_levels/
│   ├── refactoring_taxonomy.ipynb      the mapping and its three checks
│   └── refactoring_levels.yaml         the mapping itself, read by 05a
```

## Running the notebooks

The notebooks run in Google Colab, Jupyter, or an IDE such as VS Code or PyCharm. Every notebook
starts with the same setup cell, which finds the repository and picks one working folder,
`WORK`, through which the notebooks hand files to each other.

| where | `WORK` is |
|---|---|
| Jupyter or an IDE | `<repository>/work` |
| Colab | a folder on your Google Drive, so a long run survives a disconnect (`CIC_DRIVE=0` keeps it on Colab's disk) |
| anywhere | the `CIC_WORK` environment variable, if set |

**Locally**

```bash
git clone https://github.com/Binamra00/commit-intent-and-refactoring-levels
cd commit-intent-and-refactoring-levels
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt jupyterlab
jupyter lab
```

**On Colab**, open any notebook from GitHub (File → Open notebook → GitHub). The setup cell
clones the repository and mounts Drive.

### You do not have to start at the beginning

If an input is missing from `WORK`, the notebook downloads it from this study's Zenodo record or
Hugging Face repository. So each notebook can be run on its own:

| file | written by | otherwise comes from |
|---|---|---|
| `levin_dataset.pkl` | 01 | re-run 01 (a minute, no GPU) |
| `levin_columns.json`, `stage2_*_idx.npy` | 01, 02 | Zenodo |
| `levin_with_diffs.jsonl` | 03a (needs a GitHub token) | Zenodo |
| `model_attention_{42,1337,2024}.pt` | 04b (about 10 hours on an A100) | Hugging Face |
| `stage4b_results.json` | 04b | Zenodo |
| `refactoring_levels.yaml` | the taxonomy notebook | `refactoring_levels/` |
| the thesis corpus | — | Zenodo record 24182007, downloaded by 05a and the taxonomy notebook |
| `stage5a_commits.jsonl` | 05a (minutes, no GPU) | re-run 05a |
| `stage5_logits.jsonl` | 05b (hours on a GPU) | Zenodo, used automatically when no GPU is present |

### Hardware

| notebook | needs |
|---|---|
| 01, 02, 03a, 03b, 05a, taxonomy | CPU |
| 04a | a GPU for about 15 minutes of encoding |
| 04b | a large GPU; about 10 hours on an 80 GB A100 |
| 04c | a GPU for a few minutes, or a slow CPU run |
| 05b | a GPU for scoring; without one it uses the published logits |

04b measures how many file pieces per commit fit in GPU memory and caps commits there. On the
80 GB A100 used here the cap was 256 and no commit was cut; a smaller GPU gets a lower cap and
may give slightly different numbers. The cap used is recorded in `stage4b_results.json`.

Re-running 03a fetches the diffs from GitHub again. Repositories can rewrite their history, so
the published `levin_with_diffs.jsonl` is the frozen input the reported results were built from.

## Data, weights and licences

| what | where |
|---|---|
| Levin & Yehudai's labelled commits | [10.5281/zenodo.835534](https://doi.org/10.5281/zenodo.835534), downloaded by 01 |
| The five-repository corpus | [10.5281/zenodo.24182007](https://doi.org/10.5281/zenodo.24182007) |
| This study's inputs, stage outputs and `dataset_a.csv` | Zenodo: TODO |
| Trained weights | Hugging Face: TODO |

The code is under the MIT licence. The diffs and commit messages come from the projects
concerned and remain under their own licences.

## Acknowledgements

The five-repository corpus that `05a_assemble` and the taxonomy notebook read (commits, commit
messages, changed lines, refactorings, issues, pull requests, and the links between them) was mined
with [Quarry](https://github.com/Binamra00/quarry), the mining tool built for the author's M.S.
thesis, and is published as [10.5281/zenodo.24182007](https://doi.org/10.5281/zenodo.24182007).
Quarry detects refactorings with RefactoringMiner 3.1.3.

The labelled commits are Levin & Yehudai's; `03a_extract_diffs` fetched their diffs from the GitHub
API. The pretrained encoder is Microsoft's CodeBERT. The level mapping is cross-checked against the
mapping Horikawa et al. published with arXiv:2511.04824.

## Citing

If you use this work, please cite it with the metadata in `CITATION.cff`, and cite the work it
builds on:

- S. Levin and A. Yehudai. *Boosting Automatic Commit Classification Into Maintenance Activities
  By Utilizing Source Code Changes.* PROMISE 2017.
- E. Murphy-Hill, C. Parnin and A. P. Black. *How We Refactor, and How We Know It.* IEEE TSE
  38(1):5–18, 2012.
- N. Tsantalis, A. Ketkar and D. Dig. *RefactoringMiner 2.0.* IEEE TSE 48(3):930–950, 2022.
- Z. Feng et al. *CodeBERT: A Pre-Trained Model for Programming and Natural Languages.*
  Findings of EMNLP 2020.

# `scripts/`

| File         | What it does                                                                                                                                                                     |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `score.py` | Prints the strict mean, per-category means and termination profile for a `results.json`; `--compare A B` compares two runs on the task ids they share. Standard library only. |

```bash
python scripts/score.py data/simulations/<run>/results.json
```

## fetch_data.py

Downloads the domain data (tasks, database, policy, knowledge base,
user-simulator guideline) from the Hugging Face dataset
`NPCI/tau-indian-banking` into `data/`. Run it once before installing the
package or running the tests; set `HF_TOKEN` first if the dataset is gated.

```bash
python scripts/fetch_data.py            # skips the download if data is present
python scripts/fetch_data.py --force    # re-download
```

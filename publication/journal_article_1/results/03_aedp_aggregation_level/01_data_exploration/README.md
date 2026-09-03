# AEDP Aggregation Data Exploration Outputs

This folder contains dataset-preparation and audit outputs for the AEDP aggregation-level workflow. These files are not PyNNLF model experiment results. PyNNLF processed summaries should live one level up in `results/03_aedp_aggregation_level`, while figures remain in `results/03_aedp_aggregation_level/figures`.

## Main Files

| File | Purpose |
| --- | --- |
| `aedp_aggregation_sample_design.csv` | Canonical mapping from `ds25`-`ds36` to aggregation level, sample number, filename, unique-household count, and total household weight. Notebook `4_process_pynnlf_output_aedp_aggregation.ipynb` reads this file. |
| `aedp_aggregation_sample_membership_with_metadata.csv` | Chosen household/site IDs for every dataset sample, with weights and site metadata such as postcode and coordinates. Best file for inspecting which households were selected. |
| `aedp_aggregation_dataset_export_summary.csv` | Validation summary for the exported `ds25`-`ds36` CSV datasets. |
| `aedp_valid_checkpoint_sites.csv` | Valid AEDP per-household checkpoints available for sampling after notebook `2.1`. |

## Auxiliary Diagnostics

| File | Purpose |
| --- | --- |
| `aedp_148hh_recovered_site_list.csv` | Recovered 148-site AEDP cohort used as the starting point. |
| `aedp_checkpoint_selected_top_level_raw_files.csv` | The 36 top-level raw monthly files selected for `2021-07` through `2024-06`; confirms nested cloud-storage duplicate folders were not used. |
| `aedp_checkpoint_monthly_scan_summary.csv` | Month-by-month raw scan summary from notebook `2.1`. |
| `aedp_site_30min_checkpoint_summary.csv` | Full checkpoint status table for all recovered cohort sites. |
| `aedp_site_30min_checkpoint_validation.csv` | Validation checks for each written per-household 30-minute checkpoint. |
| `aedp_excluded_checkpoint_sites.csv` | Sites excluded from sampling because they did not produce usable full-period `ac_load_net` checkpoints. |
| `aedp_aggregation_sample_membership.csv` | Compact sample membership table without site metadata. |

## Producer Notebooks

- `1_build_aedp_site_30min_checkpoints.ipynb` creates checkpoint and site-audit files.
- `2_generate_aedp_aggregation_datasets.ipynb` creates sample-design, membership, and export-summary files.


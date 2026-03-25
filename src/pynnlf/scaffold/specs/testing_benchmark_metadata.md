# Testing benchmark metadata

Benchmarks were produced on 3 machines and aggregated into min/max acceptable values in `testing_benchmark.csv`.

Machines (historical):
- Dell Precision 3480 (i5-1350P, 32GB RAM)
- Dell Precision 5820 Tower (Xeon W-2175, 64GB RAM)
- Lenovo Thinkpad T480s (i7-8550U, 16GB RAM)

Interpretation:
- For each metric_id, `min acceptable value` and `max acceptable value` define the acceptable range.
- The automated test runner compares the newly produced value to these bounds and reports pass/fail (warning-only).
- The acceptable range is based on historical runs across multiple machines and may be widened for metrics that show consistent cross-machine variability.

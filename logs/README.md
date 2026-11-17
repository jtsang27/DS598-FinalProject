# logs/

Numbers recorded during training: one `metrics.csv` per run, with one row per test set at
every check (step, phase, loss, accuracy, …). Any training curve can be redrawn from it.

`metrics.csv` files are small and may be committed. Batch-job output (`logs/sge/`) and
`*.log` files stay on the SCC only.

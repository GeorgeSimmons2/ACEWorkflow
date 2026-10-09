#!/bin/bash
#SBATCH --job-name=npt_al16_lowT_done
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --output=npt_al16/logs/done_lowT_%j.log
cd /storage/astro2/phupfb/PhD/acestuff/ACEWorkflow
{ echo "# Al_16 NPT 100 K + 200 K finished $(date)"; sacct -j "$WATCH" -X -n -o JobID%14,State%12,Elapsed
  for d in npt_al16/runs_100_200/*/; do echo "## $(basename $d)"
    for T in 100 200; do cat $d/T${T}K/summary_row.csv 2>/dev/null | tail -1; done; done
} > npt_al16/RUN_100_200K_DONE.md

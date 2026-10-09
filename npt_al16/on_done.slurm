#!/bin/bash
#SBATCH --job-name=npt_al16_done
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --output=npt_al16/logs/done_%j.log
cd /storage/astro2/phupfb/PhD/acestuff/ACEWorkflow
{ echo "# Al_16 NPT 300 K finished $(date)"; sacct -j "$WATCH" -X -n -o JobID%14,State%12,Elapsed
  for d in npt_al16/runs/*/; do echo "## $(basename $d)"; cat $d/a0.csv $d/T300K/summary_row.csv 2>/dev/null; done
} > npt_al16/RUN_300K_DONE.md

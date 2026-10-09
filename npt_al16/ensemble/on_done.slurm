#!/bin/bash
#SBATCH --job-name=npt_al16_ens_done
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --time=00:15:00
#SBATCH --output=npt_al16/ensemble/logs/done_%j.log
cd /storage/astro2/phupfb/PhD/acestuff/ACEWorkflow
{ echo "# Al_16 ensemble NPT finished $(date)"; sacct -j "$WATCH" -X -n -o JobID%16,JobName%10,State%12,Elapsed
  echo; echo "summary rows present: $(ls npt_al16/ensemble/runs/*/member_*/T*K/summary_row.csv 2>/dev/null | wc -l) / 246"
} > npt_al16/ensemble/ENSEMBLE_DONE.md

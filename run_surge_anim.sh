#!/bin/bash
# run python script to generate animated map gif of latest surge forecast
# This will generate /projectsa/surge_archive/figures/surge_anom_latest.gif 
# crontab -e
# 30 19 * * * /login/jelt/GitHub/NTSLF_code/run_surge_anim.sh
# 30 07 * * * /login/jelt/GitHub/NTSLF_code/run_surge_anim.sh

source /packages/lmodmodules/apps/anaconda/5-2021/bin/activate /work/jelt/conda-env/ntslf_py39
source /etc/profile.d/modules.sh
#python Documents/my_python_file_name.py WRONG. SEPARATLY GO TO FOLDER THEN EXECUTE PYTHON
cd /login/jelt/GitHub/NTSLF_code/  
python surge_anim.py > surge_anim.log
conda deactivate

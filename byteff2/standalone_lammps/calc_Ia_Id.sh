#!/bin/bash

for i in DFETHF TFETHF PFPTHF SFBTHF
do
cd $i
echo $i
python ../ia_id_from_mdanalysis.py \
  --top transport_working_dir/solvent_salt.gro \
  --traj nvt.dcd \
  --cation-sel "resname LI" \
  --anion-sel-all "resname FSI" \
  --anion-contact-sel "resname FSI and (name O*)" \
  --cutoff 2.6 \
  --start 0 --stop -1 --step 1 \
  --out metrics.csv
cd ../
done
#python ../ia_id_from_mdanalysis.py \
#  --top config.pdb \
#  --traj nvt3.lammpsdump \
#  --cation-sel "resname Li" \
#  --anion-sel-all "resname fsa" \
#  --anion-contact-sel "resname fsa and (name O*)" \
#  --cutoff 2.5 \
#  --start 0 --stop -1 --step 1 \
#  --out metrics.csv

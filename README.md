Linna An used some old scripts, maybe some from Gyu Rie, some from Bcov, some from Hugh Huddox, and generated a script which:
1. relax pdb complex with bb cst
2. (this script from bcov)use the relaxed pdb, measure per residue ddg
3. report the all hb info

to run:
`apptainer run my_pyrosetta.sif --pdb_in {your_pdb}.pdb`
at your pdb directory, you will see:
```
{your_pdb}_relax.pdb # contains some basic rosetta scores at the end of the file
{your_pdb}_relax_perddg.json # this contains per_residue ddg
{your_pdb}_relax_hb_info.csv # this contains all detailed hb info
```

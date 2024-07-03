Linna An used some old scripts, some from Gyu Rie Lee, some from Brian Coventry, some from Hugh Haddox, and generated a script which:
1. relax pdb complex with bb cst
2. (this script from bcov)use the relaxed pdb, measure per residue ddg
3. report the all hb info (many basics come from Hugh Haddox)

This script is created to make it easy for designers to do some simple score for PPI evaluation. It should be also easy to add more filters or customized to other situations, and do later operations.

Special thanks to everyone mentioned above. 

Please cite/acknowledge the repo if you used the script for whatever use.

to run:
`apptainer run my_pyrosetta.sif --pdb_in {your_pdb}.pdb`
at your pdb directory, you will see:
```
{your_pdb}_relax.pdb # contains some basic rosetta scores at the end of the file
{your_pdb}_relax_perddg.json # this contains per_residue ddg
{your_pdb}_relax_hb_info.csv # this contains all detailed hb info
```

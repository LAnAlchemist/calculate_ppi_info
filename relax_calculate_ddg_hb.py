import os
import sys
import re
import random
import numpy as np
import pandas as pd
import math
import glob
import matplotlib
import matplotlib.pyplot as plt
import subprocess
from collections import defaultdict
import json

from pyrosetta import *
import pyrosetta.distributed.tasks.rosetta_scripts as rosetta_scripts
import pyrosetta.distributed.packed_pose as packed_pose
import argparse

init(f'-beta_nov16 -corrections:beta_nov16')

argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
argparser.add_argument("--pdb_in", type=str, default='', help="input pdb file, chA is your design, chB is your target")
args = argparser.parse_args()    

assert args.pdb_in != '', print('[ERROR]:missing input!')
    
pdb_in = args.pdb_in

pose_in = pose_from_pdb(pdb_in)
scorefxn = pyrosetta.rosetta.core.scoring.ScoreFunctionFactory.create_score_function("beta_nov16")
scorefxn(pose_in)

## quick relax
def repack_pose(xml_in):
    return rosetta_scripts.SingleoutputRosettaScriptsTask(xml_in)

XML_FASTRELAX = """
<ROSETTASCRIPTS>
  <SCOREFXNS>
    <ScoreFunction name="scorefxn_full" weights="ref2015">
      <Reweight scoretype="coordinate_constraint" weight="0.5"/>
      <Reweight scoretype="atom_pair_constraint" weight="0.1" />
    </ScoreFunction>
    <ScoreFunction name="scorefxn_soft" weights="ref2015">
      <Reweight scoretype="coordinate_constraint" weight="1.0"/>
      <Reweight scoretype="fa_rep" weight="0.2"/>
      <Reweight scoretype="atom_pair_constraint" weight="0.1" />
    </ScoreFunction>
  </SCOREFXNS>

  <RESIDUE_SELECTORS>
    <Chain name="chA" chains="A"/>
    <Chain name="chB" chains="B"/>
    <Or name="all" selectors="chA,chB"/>
    <InterfaceByVector name="interface_by_vector" cb_dist_cut="11" nearby_atom_cut="5.5" vector_angle_cut="75" vector_dist_cut="9" grp1_selector="chA" grp2_selector="chB"/>
    <Not name="fix_res" selector="interface_by_vector"/>
  </RESIDUE_SELECTORS>

  <TASKOPERATIONS>
    <OperateOnResidueSubset name="dont_repack" selector="fix_res">
      <PreventRepackingRLT/>
    </OperateOnResidueSubset>
    <OperateOnResidueSubset name="repack_selected_res" selector="interface_by_vector">
      <RestrictToRepackingRLT/>
    </OperateOnResidueSubset>
    <InitializeFromCommandline name="init"/>
    <ExtraRotamersGeneric name="ex1ex2" ex1="1" ex2="1"/>
  </TASKOPERATIONS>

  <FILTERS>
    <ScoreType name="totalscore" scorefxn="scorefxn_full" threshold="9999" confidence="0"/>
    <ResidueCount name="nres" confidence="0" />
    <CalculatorFilter name="res_totalscore" confidence="0" equation="SCORE/NRES" threshold="999">
      <Var name="SCORE" filter_name="totalscore" />
      <Var name="NRES" filter_name="nres" />
    </CalculatorFilter>
    /assume binder at chain A
    <Ddg name="ddg" threshold="99999" jump="1" repeats="1" repack="1" repack_bound="true" repack_unbound="true" relax_unbound="false" scorefxn="scorefxn_full"/>
    <ContactMolecularSurface name="cms" target_selector="chB" binder_selector="chA" use_rosetta_radii="true"/>
  </FILTERS>

  <MOVERS>
    <AddConstraints name="add_ca_csts" >
      <CoordinateConstraintGenerator name="coord_cst_gen" ca_only="true"/>
    </AddConstraints>
    ConstraintSetMover name="add_lig_d_csts" add_constraints="1" cst_file="1"/>
    ConstraintSetMover name="rm_lig_d_csts" cst_file="none"/>
    <ClearConstraintsMover name="rm_all_cst" />
    <RemoveConstraints name="rm_csts" constraint_generators="coord_cst_gen"/>

    <FastRelax name="fastrelax" scorefxn="scorefxn_full" task_operations="init,ex1ex2,repack_selected_res,dont_repack" repeats="1" /> #cst_file="1" 
  </MOVERS>

  <PROTOCOLS>
    <Add mover="add_ca_csts"/>
    Add mover="add_lig_d_csts"/>
    <Add mover="fastrelax"/>
    Add mover="rm_csts"/>
    Add mover="rm_lig_d_csts"/>
    <Add mover="rm_all_cst"/>
    <Add filter="ddg"/>
    <Add filter="cms"/>
    <Add filter="res_totalscore"/>
    <Add filter="totalscore"/>
  </PROTOCOLS>

</ROSETTASCRIPTS>

"""


# report per_res ddg
def move_chainA_far_away(pose):
    pose = pose.clone()
    sel = pyrosetta.rosetta.core.select.residue_selector.ChainSelector("A")
    subset = sel.apply(pose)

    x_unit = pyrosetta.rosetta.numeric.xyzVector_double_t(1, 0, 0)
    far_away = pyrosetta.rosetta.numeric.xyzVector_double_t(10000, 0, 0)

    pyrosetta.rosetta.protocols.toolbox.pose_manipulation.rigid_body_move(x_unit, 0, far_away, pose, subset)

    return pose

def main():
    xml_obj = repack_pose(XML_FASTRELAX)
    xml_obj.setup()
    pose_work = xml_obj.apply(pose_in)
    out_pdb = pdb_in[:-4]+'_relax.pdb'
    packed_pose.to_pose(pose_work).dump_pdb(out_pdb)

    ## read in relax_pose
    relax_pose = pose_from_pdb(out_pdb)

    scorefxn(relax_pose)

    sep = move_chainA_far_away(relax_pose)
    scorefxn(sep)

    per_res_ddg = {}
    for i in range(1, relax_pose.size()+1):
        num = relax_pose.pdb_info().number(i)

        together = relax_pose.energies().residue_total_energy(i)
        apart = sep.energies().residue_total_energy(i)
        per_res_ddg[num]=[relax_pose.residue(i).name1(),i,num,(together-apart)]

        print("%s %3i %3i %6.1f   %6.1f - %6.1f"%(relax_pose.residue(i).name1(), i, num, 2*(together - apart), together, apart))

    with open(out_pdb[:-4]+'_perddg.json','w') as fp:
        json.dump(per_res_ddg,fp)
        
    from hbond_utils import compute_degree_satisfaction
    degree_sat_df = compute_degree_satisfaction(out_pdb)
    degree_sat_df.to_csv(out_pdb[:-4]+'_hb_info.csv')
    
if __name__ == '__main__':
    main()
        
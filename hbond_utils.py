"""
This script contains various modules for analyzing hydrogen bonds using PyRosetta

Hugh Haddox, March 27, 2019
"""

# Import `Python` modules
import os
import pandas
import numpy as np
# from Bio.Alphabet import IUPAC

# Import and initialize `PyRosetta`
import pyrosetta
import pyrosetta.rosetta
from pyrosetta.rosetta.core.scoring.hbonds import HBondSet, fill_hbond_set
pyrosetta.init('-beta_nov16 -corrections:beta_nov16 -mute all -mute core -mute protocols -extra_res_fa /home/linnaan/ligands/FOL/FOL_classic_redo/FOL.params') #LA: remove -read_only_ATOM_entries, which only read ATOM entries 

# Set up a score function in `PyRosetta`
sf = pyrosetta.get_fa_scorefxn()
def fix_scorefxn(sfxn, allow_double_bb=False):
    opts = sfxn.energy_method_options()
    opts.hbond_options().decompose_bb_hb_into_pair_energies(True)
    opts.hbond_options().bb_donor_acceptor_check(not allow_double_bb)
    sfxn.set_energy_method_options(opts)
fix_scorefxn(sf)

def find_hbonds(p, derivatives=False, exclude_bb=True, exclude_bsc=True,
                exclude_scb=True, exclude_sc=False):
    """Find all hydrogen bonds of a particular type in a supplied Pose
    and return them as a HBondSet.

    Args:
        p (pyrosetta.Pose): The Pose from which to extract hydrogen
            bonds.
        derivatives (bool, optional): Evaluate energy derivatives and
            store them in the HBondSet. Defaults to False.
        exclude_bb (bool, optional): If True, do not store
            backbone--backbone hydrogen bonds in the HBondSet. Defaults
            to True.
        exclude_bsc (bool, optional): If True, do not store
            backbone--side chain hydrogen bonds in the HBondSet.
            Defaults to True.
        exclude_scb (bool, optional): If True, do not store
            side chain--backbone hydrogen bonds in the HBondSet.
            Defaults to True.
        exclude_sc (bool, optional): If True, do not store
            side chain--side chain hydrogen bonds in the HBondSet.
            Defaults to False.

    Returns:
        pyrosetta.rosetta.core.scoring.hbonds.HBondSet: A hydrogen bond
        set containing the specified types of hydrogen bonds in the
        Pose.
    """
    p.update_residue_neighbors()
    hbset = HBondSet()
    fill_hbond_set(
        p, derivatives, hbset, exclude_bb, exclude_bsc, exclude_scb,
        exclude_sc
    )
    return hbset

# Define functions to import in other scripts
def compute_degree_satisfaction(pdb, vsasa_burial_cutoff=0.1):
    "Compute the degree that each polar atom is satisfied in an input PDB"

    # Read in and score the pose
    pose = pyrosetta.pose_from_pdb(pdb)
    sf(pose)

    # Compute the VSASA of every atom in the pose
    sasa_calc = \
        pyrosetta.rosetta.protocols.vardist_solaccess.VarSolDistSasaCalculator()
    sasa_map = sasa_calc.calculate(pose)

    # Get an object with all hydrogen bonds in pose
    hbond_set = pose.get_hbonds()

    # Cycle through each residue in the pose, and each atom
    # in a given residue. Look to see if the atom is buried,
    # whether or not it is capable of making H bonds, and if so,
    # how many H bonds it is making
    degree_satisfaction_dict = {
        key : []
        for key in [
            'res_n', 'res_name', 'atom_n', 'atom_name', 'atom_type_name',
            'atom_type_element', 'buried', 'heavy_atom_acceptor',
            'heavy_atom_donor', 'n_hbonds_as_acceptor', 'n_hpols',
            'n_hpols_making_hbonds', 'n_hpol_hbonds'
        ]
    }
    for res_n in list(range(1, pose.size()+1)):
        res = pose.residue(res_n)
        for atom_n in list(range(1, res.natoms()+1)):

            # Only continue if the atom is a heavy-atom and
            # is an hbond donor and/or acceptor
            atom_type = res.atom_type(atom_n)
            if not atom_type.is_heavyatom():
                continue
            heavy_atom_is_an_acceptor = atom_type.is_acceptor()
            heavy_atom_is_a_donor = atom_type.is_donor()
            if (not heavy_atom_is_an_acceptor) and \
                (not heavy_atom_is_a_donor):
                continue

            # Get the AtomID, hydrogen bonds, and SASA for
            # the atom in question
            atom_id = pyrosetta.AtomID(atom_n, res_n)
            atom_hbonds = hbond_set.atom_hbonds(
                atom_id, include_only_allowed=False
            )
            atom_sasa = sasa_map[res_n][atom_n]

            # Add metadata to dictionary
            degree_satisfaction_dict['res_n'].append(res_n)
            degree_satisfaction_dict['res_name'].append(res.name1())
            degree_satisfaction_dict['atom_n'].append(atom_n)
            degree_satisfaction_dict['atom_name'].append(
                res.atom_name(atom_n).strip()
            )
            degree_satisfaction_dict['atom_type_name'].append(
                atom_type.atom_type_name()
            )
            degree_satisfaction_dict['atom_type_element'].append(
                atom_type.element()
            )
            degree_satisfaction_dict['heavy_atom_acceptor'].append(
                heavy_atom_is_an_acceptor
            )
            degree_satisfaction_dict['heavy_atom_donor'].append(
                heavy_atom_is_a_donor
            )

            # Get the number of hbonds made as an acceptor
            if heavy_atom_is_an_acceptor:
                degree_satisfaction_dict['n_hbonds_as_acceptor'].append(
                    len(atom_hbonds)
                )
            else:
                degree_satisfaction_dict['n_hbonds_as_acceptor'].append(
                    np.nan
                )

            # If the atom is a donor, count how many polar hydrogens
            # it has, how many of them are making hbonds, and how many
            # of them are both unsatisfied AND buried. Note: burial is
            # computed for the heavy-atom donor, not the polar hydrogen
            # in question, to be consistent with the unsat filter
            if heavy_atom_is_a_donor:

                # Do the above by cycling through each bonded neighboring
                # atom and analyzing the polar hydrogens
                bonded_neighbor_atoms = res.bonded_neighbor(atom_n)
                n_hpols = 0
                n_hpol_hbonds = 0
                n_hpols_making_hbonds = 0
                for neighbor_atom_n in bonded_neighbor_atoms:
                    if res.atom_is_polar_hydrogen(neighbor_atom_n):

                        # Record info about the polar hydrogen in question
                        n_hpols += 1
                        hpol_atom_id = pyrosetta.AtomID(
                            neighbor_atom_n, res_n
                        )
                        n_hpol_atom_hbonds = len(hbond_set.atom_hbonds(
                            hpol_atom_id, include_only_allowed=False
                        ))
                        n_hpol_hbonds += n_hpol_atom_hbonds
                        if n_hpol_atom_hbonds > 0:
                            n_hpols_making_hbonds += 1

                        # Add the SASA of the polar hydrogen
                        # to the SASA of the heavy atom plus any
                        # other polar hydrogens considered so far
                        hpol_atom_sasa = sasa_map[res_n][neighbor_atom_n]
                        atom_sasa += hpol_atom_sasa

                # Add data to dictionary
                degree_satisfaction_dict['n_hpols'].append(n_hpols)
                degree_satisfaction_dict['n_hpols_making_hbonds'].append(
                    n_hpols_making_hbonds
                )
                degree_satisfaction_dict['n_hpol_hbonds'].append(
                    n_hpol_hbonds
                )

            # If the atom in question is not a donor atom, then
            # just fill in the remaining dictionary entries with
            # values of nan
            else:
                keys = [
                    'n_hpols', 'n_hpols_making_hbonds',
                    'n_hpol_hbonds'
                ]
                for key in keys:
                    degree_satisfaction_dict[key].append(np.nan)

            # Determine whether the atom is buried now that the
            # SASA of all heavy atoms and polar hydrogens has been
            # taken into account
            atom_is_buried = atom_sasa < vsasa_burial_cutoff
            degree_satisfaction_dict['buried'].append(
                atom_is_buried
            )

    degree_sat_df = pandas.DataFrame.from_dict(
        degree_satisfaction_dict
    )

    # Add additional columns to the dataframe
    degree_sat_df['bb_or_sc'] = degree_sat_df.apply(
        lambda row: 'bb' if pose.residue(row['res_n']).atom_is_backbone(
            row['atom_n']
        ) else 'sc', axis=1
    )
    degree_sat_df['heavy_atom_unsat'] = degree_sat_df.apply(
        lambda row: \
            (
                (row['heavy_atom_acceptor'] == True) &
                (row['heavy_atom_donor'] == False) &
                (row['n_hbonds_as_acceptor'] == 0)
            ) |
            (
                (row['heavy_atom_acceptor'] == False) &
                (row['heavy_atom_donor'] == True) &
                (row['n_hpols_making_hbonds'] == 0)
            ) |
            (
                (row['heavy_atom_acceptor'] == True) &
                (row['heavy_atom_donor'] == True) &
                (row['n_hbonds_as_acceptor'] == 0) &
                (row['n_hpols_making_hbonds'] == 0)
            ),
            axis=1
    )
    degree_sat_df['heavy_atom_buried_unsat'] = \
        degree_sat_df.apply(
            lambda row: row['heavy_atom_unsat'] & row['buried'],
            axis=1
        )
    degree_sat_df['bb_buried_unsat'] = \
        degree_sat_df.apply(
            lambda row: \
                (row['heavy_atom_buried_unsat']) &
                (row['bb_or_sc'] == 'bb'),
            axis=1
        )
    degree_sat_df['sc_buried_unsat'] = \
        degree_sat_df.apply(
            lambda row: \
                (row['heavy_atom_buried_unsat']) &
                (row['bb_or_sc'] == 'sc'),
            axis=1
        )
    return degree_sat_df

def compute_percent_capacity_of_atom(capacity, n_observed_hbonds):
    percent_capacity = 100 * (1 - (
        max(0, ((capacity - n_observed_hbonds)) / capacity)
    ))
    return percent_capacity

def determine_if_res_has_available_hbond_don_and_acc(
    df, acceptor_hbond_capacities_dict=False
):
    """
    Determine if a residue has an available H-bond donor or acceptor

    Args:
        *df*: a dataframe from *compute_degree_satisfaction*
    Returns:
        A tuple of two bools indicating whether the residue has available
            an available acceptor and/or an available donor, respectively

    """

    # A dictionary defining the capacity of heavy-atom h-bond acceptors
    # where keys are `atom_type_name` variables and values are integers
    # giving the atom's capacity. The capacity of heavy-atom donors is
    # simply defined as the number of Hpols bonded to that atom
    if not acceptor_hbond_capacities_dict:
        acceptor_hbond_capacities_dict = {
            ('OH') : 2,
            ('OOC') : 2,
            ('ONH2') : 2,
            ('Nhis') : 1
        }

    # Loop over every atom in the input dataframe and record the degree
    # that it is satisfied
    available_acc = False
    available_don = False
    for (i, row) in df.iterrows():

        # If the atom is an acceptor, record the degree that it is
        # satisfied in its capacity as an acceptor
        if row['heavy_atom_acceptor']:
            acc_capacity = acceptor_hbond_capacities_dict[row['atom_type_name']]
            n_observed_hbonds = row['n_hbonds_as_acceptor']
            acc_degree_sat = compute_percent_capacity_of_atom(
                acc_capacity,
                n_observed_hbonds
            )
            if acc_degree_sat < 100:
                available_acc = True

        # If the atom is a donor, record the degree that it is
        # satisfied in its capacity as a donor
        if row['heavy_atom_donor']:
            don_capacity = row['n_hpols']
            n_hpols_making_hbonds = row['n_hpols_making_hbonds']
            don_degree_sat = compute_percent_capacity_of_atom(
                don_capacity,
                n_hpols_making_hbonds
            )
            if don_degree_sat < 100:
                available_don = True

    return (available_acc, available_don)

def compute_percent_capacity_summary_statistics(df, acceptor_hbond_capacities_dict=False):
    """Compute summary statistics on the percent capacity of all input atoms"""

    # A dictionary defining the capacity of heavy-atom h-bond acceptors
    # where keys are `atom_type_name` variables and values are integers
    # giving the atom's capacity. The capacity of heavy-atom donors is
    # simply defined as the number of Hpols bonded to that atom
    if not acceptor_hbond_capacities_dict:
        acceptor_hbond_capacities_dict = {
            ('OH') : 2,
            ('OOC') : 2,
            ('ONH2') : 2,
            ('Nhis') : 1
        }

    # Loop over every atom in the input dataframe and record the degree
    # that it is satisfied
    acc_num_hbonds = {}
    don_num_hpols_sat = {}
    acc_degree_sats = []
    don_degree_sats = []
    for (i, row) in df.iterrows():

        # If the atom is an acceptor, record the degree that it is
        # satisfied in its capacity as an acceptor
        if row['heavy_atom_acceptor']:
            acc_capacity = acceptor_hbond_capacities_dict[row['atom_type_name']]
            n_observed_hbonds = row['n_hbonds_as_acceptor']
            acc_degree_sat = compute_percent_capacity_of_atom(
                acc_capacity,
                n_observed_hbonds
            )
            acc_degree_sats.append(acc_degree_sat)
            sat_group = '_'.join([
                'acc', row['res_name'], row['atom_type_name'],
                str(int(n_observed_hbonds))
            ])
            if sat_group not in acc_num_hbonds.keys():
                acc_num_hbonds[sat_group] = 1
            else:
                acc_num_hbonds[sat_group] += 1
        else:
            (acc_capacity, acc_degree_sat) = (None, None)

        # If the atom is a donor, record the degree that it is
        # satisfied in its capacity as a donor
        if row['heavy_atom_donor']:
            don_capacity = row['n_hpols']
            n_hpols_making_hbonds = row['n_hpols_making_hbonds']
            don_degree_sat = compute_percent_capacity_of_atom(
                don_capacity,
                n_hpols_making_hbonds
            )
            don_degree_sats.append(don_degree_sat)
            sat_group = '_'.join([
                'don', row['res_name'], row['atom_type_name'],
                str(int(n_hpols_making_hbonds))
            ])
            if sat_group not in don_num_hpols_sat.keys():
                don_num_hpols_sat[sat_group] = 1
            else:
                don_num_hpols_sat[sat_group] += 1
        else:
            (don_capacity, don_degree_sat) = (None, None)

    # Record data for output
    output_dict = {
        'avg_acc_percent_capacity' : [np.mean(acc_degree_sats)],
        'avg_don_percent_capacity' : [np.mean(don_degree_sats)],
        'avg_percent_capacity' : [np.mean(acc_degree_sats + don_degree_sats)]
    }
    for (sat_group, n_hbonds) in acc_num_hbonds.items():
        output_dict[sat_group] = n_hbonds
    for (sat_group, n_hbonds) in don_num_hpols_sat.items():
        output_dict[sat_group] = n_hbonds
    output_df = pandas.DataFrame.from_dict(output_dict)

    return output_df

def determine_residue_satisfaction(res_df):
    """
    Compute the degree that an entire residue is satisfied
    """

    # Iterate through each atom in the input data for
    # a given residue
    res_name = set(res_df['res_name'])
    assert len(res_name) == 1
    res_name = list(res_name)[0]
    data_on_res = []
    for (i, row) in res_df.iterrows():
        atom_type_name = row['atom_type_name']
        if row['heavy_atom_acceptor']:
            data_on_res.append('_'.join([
                row['atom_type_name'], 'acc_{0}'.format(
                    int(row['n_hbonds_as_acceptor'])
                )
            ]))
        if row['heavy_atom_donor']:
            data_on_res.append('_'.join([
                row['atom_type_name'], 'don_{0}'.format(
                    int(row['n_hpols_making_hbonds'])
                )
            ]))
    data_on_res = '__'.join([res_name] + sorted(data_on_res))

    return data_on_res

def compute_hbond_energies(
    pdb, vsasa_cutoff=0.1, only_return_buried_hbonds=True,
    exclude_bb=True, exclude_bsc=False, exclude_scb=False, exclude_sc=False
):

    # Read in and score pose
    pose = pyrosetta.pose_from_pdb(pdb)
    sf(pose)

    # Compute the VSASA of every atom in the pose
    sasa_calc = \
        pyrosetta.rosetta.protocols.vardist_solaccess.VarSolDistSasaCalculator()
    sasa_map = sasa_calc.calculate(pose)

    hbset = find_hbonds(
        pose,
        exclude_bb=exclude_bb,
        exclude_bsc=exclude_bsc,
        exclude_scb=exclude_scb,
        exclude_sc=exclude_sc
    )
    if len(hbset.hbonds()) == 0:
        return None

    hbond_dict = {
        key : []
        for key in [
            'pdb',

            'acc_res_n', 'acc_atom_n', 'acc_res_name', 'acc_atom_name',
            'acc_atom_type_name', 'acc_vsasa', 'acc_hybridization',

            'don_res_n', 'don_atom_n', 'don_res_name', 'don_atom_name',
            'don_atom_type_name', 'don_heavy_vsasa',

            'buried', 'energy', 'weight', 'weighted_energy',
            'HAdist', 'AHDangle', 'BAHangle', 'BAtorsion',

            'lj_atr', 'lj_rep', 'fa_solv', 'fa_elec',
            'energy_hbond_plus_elec', 'weighted_energy_hbond_plus_elec'
        ]
    }

    for hb in hbset.hbonds():

        # Add metadata on PDB
        hbond_dict['pdb'].append(pdb)

        # Get metadata on the acceptor atom
        acc_res_n = hb.acc_res()
        acc_atom_n = hb.acc_atm()
        acc_res = pose.residue(acc_res_n)
        acc_res_name = acc_res.name3()
        acc_atom_name = acc_res.atom_name(acc_atom_n)
        acc_atom_type_name = \
            acc_res.atom_type(acc_atom_n).atom_type_name()
        acc_vsasa = sasa_map[acc_res_n][acc_atom_n]
        acc_hybridization = \
            acc_res.atom_type(acc_atom_n).hybridization()

        hbond_dict['acc_res_n'].append(acc_res_n)
        hbond_dict['acc_atom_n'].append(acc_atom_n)
        hbond_dict['acc_res_name'].append(acc_res_name)
        hbond_dict['acc_atom_name'].append(acc_atom_name)
        hbond_dict['acc_atom_type_name'].append(acc_atom_type_name)
        hbond_dict['acc_vsasa'].append(acc_vsasa)
        hbond_dict['acc_hybridization'].append(acc_hybridization)

        # ... and for the donor residue/atom
        don_res_n = hb.don_res()
        don_atom_n = hb.don_hatm()
        don_res = pose.residue(don_res_n)
        don_res_name = don_res.name3()
        don_atom_name = don_res.atom_name(don_atom_n)
        don_atom_type_name = don_res.atom_type(don_atom_n).atom_type_name()

        # Identify the heavy atom donating the polar hydrogen
        # and compute its SASA by summing the SASA of that atom and
        # the SASA of all all attached polar hydrogens
        neighbor_atoms = don_res.bonded_neighbor(don_atom_n)
        assert len(neighbor_atoms) == 1
        don_heavy_atom_n = list(neighbor_atoms)[0]
        don_heavy_vsasa = sasa_map[don_res_n][don_heavy_atom_n]
        bonded_neighbor_atoms = don_res.bonded_neighbor(don_heavy_atom_n)
        for neighbor_atom_n in bonded_neighbor_atoms:
            if don_res.atom_is_polar_hydrogen(neighbor_atom_n):
                hpol_atom_id = pyrosetta.AtomID(
                    neighbor_atom_n, don_res_n
                )
                hpol_atom_sasa = sasa_map[don_res_n][neighbor_atom_n]
                don_heavy_vsasa += hpol_atom_sasa

        hbond_dict['don_res_n'].append(don_res_n)
        hbond_dict['don_atom_n'].append(don_atom_n)
        hbond_dict['don_res_name'].append(don_res_name)
        hbond_dict['don_atom_name'].append(don_atom_name)
        hbond_dict['don_atom_type_name'].append(don_atom_type_name)
        hbond_dict['don_heavy_vsasa'].append(don_heavy_vsasa)

        # Determine whether the H-bond is buried based on the cutoff
        # from above
        buried = (
            (acc_vsasa < vsasa_cutoff) &
            (don_heavy_vsasa < vsasa_cutoff)
        )
        hbond_dict['buried'].append(buried)

        # Get data on the hbond energy, distance, bond angles
        energy = hb.energy()
        weight = hb.weight()
        HAdist = hb.get_HAdist(pose)
        AHDangle = hb.get_AHDangle(pose)
        BAHangle = hb.get_BAHangle(pose)
        BAtorsion = hb.get_BAtorsion(pose)

        hbond_dict['energy'].append(energy)
        hbond_dict['weight'].append(weight)
        hbond_dict['weighted_energy'].append(energy*weight)
        hbond_dict['HAdist'].append(HAdist)
        hbond_dict['AHDangle'].append(AHDangle)
        hbond_dict['BAHangle'].append(BAHangle)
        hbond_dict['BAtorsion'].append(BAtorsion)

        # Get data on other pairwise energies between the donor and acceptor
        # atom, including lj_atr, lj_rep, fa_solv, and fa_elec
        etable_atom_pair_energies = \
            pyrosetta.toolbox.atom_pair_energy.etable_atom_pair_energies
        (lj_atr, lj_rep, fa_solv, fa_elec) = etable_atom_pair_energies(
            res1=acc_res,
            atom_index_1=acc_atom_n,
            res2=don_res,
            atom_index_2=don_atom_n,
            sfxn=sf
        )
        hbond_dict['lj_atr'].append(lj_atr)
        hbond_dict['lj_rep'].append(lj_rep)
        hbond_dict['fa_solv'].append(fa_solv)
        hbond_dict['fa_elec'].append(fa_elec)
        hbond_dict['energy_hbond_plus_elec'].append(energy+fa_elec)
        hbond_dict['weighted_energy_hbond_plus_elec'].append((energy+fa_elec)*weight)

    hbond_df = pandas.DataFrame.from_dict(hbond_dict)

    # If specified, subset the dataframe to only include H-bonds that are
    # buried
    if only_return_buried_hbonds:
        hbond_df = hbond_df[hbond_df['buried'] == True]

    return hbond_df


def compute_residue_burial(pdb, sequence=False, aas_to_analyze=False):
    """
    Compute per-residue SASA values using all side-chain atoms

    Args:
        *pdb*: the path to a PDB file
        *aas_to_analyze*: by default, this is turned off. But if
            passed a list of amino acids (single-letter codes), it
            will only consider the amino-acids in question when
            computing per-residue SASA values
    Returns:
        A dataframe with the following columns, sorted from low-to
        high values of SASA:
            *res_n*: residue number
            *aa*: amino acid
            *sasa*: the residue's SASA, summed over all side-chain
                atoms
    """

    # Read in and score the pose
    if sequence:
        pose = pyrosetta.pose_from_sequence(pdb)
    else:
        pose = pyrosetta.pose_from_pdb(pdb)
    sf = pyrosetta.get_fa_scorefxn()
    sf(pose)

    # Compute the VSASA of every atom in the pose
    sasa_calc = \
        pyrosetta.rosetta.protocols.vardist_solaccess.VarSolDistSasaCalculator()
    sasa_map = sasa_calc.calculate(pose)


    # For each residue, record the SASA of all side-chain atoms,
    # only considering hydrophobic residues (excluding alanine)
    sasa_dict = {
        key : []
        for key in ['res_n', 'aa', 'sasa']
    }

    for res_n in list(range(1, pose.size()+1)):
        res = pose.residue(res_n)
        aa = res.name1()
        if aas_to_analyze:
            if aa not in aas_to_analyze:
                continue
        res_sasa = 0
        for atom_n in list(range(1, res.natoms()+1)):
            if res.atom_is_backbone(atom_n):
                continue
            res_sasa += sasa_map[res_n][atom_n]
        sasa_dict['res_n'].append(res_n)
        sasa_dict['aa'].append(aa)
        sasa_dict['sasa'].append(res_sasa)

    # Convert data to dataframe, sort rows by SASA values,
    # and re-index. Then return the dataframe
    sasa_df = pandas.DataFrame(sasa_dict)
    sasa_df.sort_values('sasa', ascending=True, inplace=True)
    sasa_df.reset_index(inplace=True, drop=True)
    return sasa_df


def compute_relative_solvent_accessibility(pdb):
    """
    Compute the relative solvent accessibility of each residue in a pose

    Args:
        *pdb*: the path to an input PDB file

    Returns:
        A dataframe that has the same info as the one returned by
            `compute_residue_burial` from above, but with an
            additional column called 'rsa' that gives per-site values
            of relative solvent accessibility
    """

    # First, get the max SASA value for each amino acid by computing
    # SASA values from 1-residue poses
    amino_acids = IUPAC.IUPACProtein.letters
    amino_acids = list(amino_acids)
    assert len(amino_acids) == 20
    dfs = []
    for aa in amino_acids:
        df = compute_residue_burial(aa, sequence=True)
        dfs.append(df)
    max_sasa_df = pandas.concat(dfs)
    max_sasa_df.set_index('aa', verify_integrity=True, inplace=True)

    # Next, compute per-residue SASA values for an input PDB
    sasa_df = compute_residue_burial(pdb)

    # Next compute RSA values
    sasa_df['rsa'] = sasa_df.apply(
        lambda row: 0.0 if row['aa'] == 'G' \
            else row['sasa'] / max_sasa_df.loc[row['aa']]['sasa'],
        axis=1
    )
    return sasa_df


def get_sf_weights():

    score_types = list(sf.get_nonzero_weighted_scoretypes())

    return {
        str(score_term).replace('ScoreType.', '') : sf.weights().get(score_term)
        for score_term in score_types
    }


def compute_per_residue_energies(pdb):
    """
    Compute per-residue energies for each term in the score function

    Args:
        *pdb*: the path to an input PDB file

    Returns:
        A dataframe with columns giving energies and rows giving
            residues
    """

    # Make a list of all score terms in the energy function
    score_types = list(sf.get_nonzero_weighted_scoretypes())
    score_terms = [
            str(score_term).replace('ScoreType.', '')
            for score_term in score_types
        ] + ['total_score']

    # Read in and score pose
    pose = pyrosetta.pose_from_pdb(pdb)
    sf_cart(pose)

    # Get weights for each score term
    weights = get_sf_weights()
    weights['total_score'] = 1.0

    # Make a dataframe with per-residue weighted scores
    scores_dict = {
        key : []
        for key in ['res_n', 'res_aa'] + score_terms
    }
    for res_n in list(range(1, pose.size()+1)):
        scores_dict['res_n'].append(res_n)
        scores_dict['res_aa'].append(pose.residue(res_n).name1())
        for score_term in score_terms:
            scores_dict[score_term].append(
                weights[score_term] * pose.energies().residue_total_energies(res_n)[
                    pyrosetta.rosetta.core.scoring.score_type_from_name(score_term)
                ]
            )

    scores_df = pandas.DataFrame(scores_dict)
    return scores_df

def compute_total_score(pdb):
    pose = pyrosetta.pose_from_pdb(pdb)
    sf(pose)
    return pyrosetta.rosetta.core.pose.total_energy_from_pose(pose)


def get_res_labels_from_pdb(pdb):
    """
    Get residue-level labels from a PDB file
    
    Args:
        *pdb*: the path to an input PDB file
    
    Returns:
        A dataframe with an entry for every residue/label combo:
            *res_n*: the residue number
            *label*: the label for that residue
            
    """
    # Read in pose
    pose = pyrosetta.pose_from_pdb(pdb)

    # Get any PDBinfo that is present for the pose. Then, for each type
    # of label, make a list of residues with that label. Do this by
    # iterating over all residues and recording things as you go.
    pdb_info = pose.pdb_info()
    labels_dict = {
        key : []
        for key in ['res_n', 'label']
    }
    for res_n in range(1, pose.size()+1):
        labels = list(pdb_info.get_reslabels(res=res_n))
        for label in labels:
            labels_dict['res_n'].append(res_n)
            labels_dict['label'].append(label)

    labels_df = pandas.DataFrame(labels_dict)
    return labels_df


def compute_avg_dist(pdb, residues):
    pose = pyrosetta.pose_from_pdb(pdb)
    ds = []
    for (i, res_i) in enumerate(residues):
        for (j, res_j) in enumerate(residues):
            if i >= j:
                continue
            xyz_i = pose.residue(res_i).xyz("CA")
            xyz_j = pose.residue(res_j).xyz("CA")
            d = (xyz_i - xyz_j).norm()
            ds.append(d)
    return np.mean(ds)


if __name__ == "__main__":
    import doctest
    doctest.testmod()

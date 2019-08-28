import scipy
from sklearn import preprocessing
from sklearn.decomposition import PCA
import networkx as nx
from collections import defaultdict
from common import *

from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem, ChemicalFeatures, rdMolTransforms
from rdkit.Chem.rdmolops import SanitizeFlags
import rdkit.Chem.Draw
from rdkit.Chem.Draw.MolDrawing import MolDrawing, DrawingOptions

DrawingOptions.bondLineWidth = 1.8

## feature extraction #####################################################

COUPLING_TYPE_STATS = [
    # type   #mean, std, min, max
    '1JHC',  94.9761528641869,   18.27722399839607,   66.6008,   204.8800,
    '2JHC',  -0.2706244378832,    4.52360876732858,  -36.2186,    42.8192,
    '3JHC',   3.6884695895355,    3.07090647005439,  -18.5821,    76.0437,
    '1JHN',  47.4798844844683,   10.92204561670947,   24.3222,    80.4187,
    '2JHN',   3.1247536134185,    3.67345877025737,   -2.6209,    17.7436,
    '3JHN',   0.9907298624944,    1.31538940138001,   -3.1724,    10.9712,
    '2JHH', -10.2866051639817,    3.97960190019757,  -35.1761,    11.8542,
    '3JHH',   4.7710233597359,    3.70498129755812,   -3.0205,    17.4841,
]
NUM_COUPLING_TYPE = len(COUPLING_TYPE_STATS)//5

COUPLING_TYPE_MEAN = [COUPLING_TYPE_STATS[i*5+1]
                      for i in range(NUM_COUPLING_TYPE)]
COUPLING_TYPE_STD = [COUPLING_TYPE_STATS[i*5+2]
                     for i in range(NUM_COUPLING_TYPE)]
COUPLING_TYPE = [COUPLING_TYPE_STATS[i*5] for i in range(NUM_COUPLING_TYPE)]

# ---

SYMBOL = ['H', 'C', 'N', 'O', 'F']

R = {'H':0.38,'C':0.77,'N':0.75,'O':0.73,'F':0.71}
E = {'H':2.2,'C':2.55,'N':3.04,'O':3.44,'F':3.98}

BOND_TYPE = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
HYBRIDIZATION = [
    # Chem.rdchem.HybridizationType.S,
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    # Chem.rdchem.HybridizationType.SP3D,
    # Chem.rdchem.HybridizationType.SP3D2,
]


def one_hot_encoding(x, set):
    one_hot = [int(x == s) for s in set]
    return one_hot


def compute_kaggle_metric(predict, coupling_value, coupling_type):
    mae = [None]*NUM_COUPLING_TYPE
    log_mae = [None]*NUM_COUPLING_TYPE
    diff = np.fabs(predict-coupling_value)
    for t in range(NUM_COUPLING_TYPE):
        index = np.where(coupling_type == t)[0]
        if len(index) > 0:
            m = diff[index].mean()
            log_m = np.log(m+1e-8)
            mae[t] = m
            log_mae[t] = log_m

    return mae, log_mae


def filter_dataframes(df, coupling_types=None):
    if coupling_types is not None:
        return df.loc[df['type'].isin(coupling_types)]
    return df


def load_csv(normalize_target=False, coupling_types=['1JHC', '2JHC', '3JHC', '1JHN', '2JHN', '3JHN', '2JHH', '3JHH']):
    DATA_DIR = get_data_path()

    # structure
    structure = pd.read_csv(DATA_DIR + 'structures.csv')
    yukawa = pd.read_csv(DATA_DIR + 'external_data/structures_yukawa.csv').fillna(0)

    df_structure = pd.concat([structure, yukawa], axis=1)

    df_train = pd.read_csv(DATA_DIR + 'train.csv')
    df_test = pd.read_csv(DATA_DIR + 'test.csv')

    df_train = filter_dataframes(df_train, coupling_types=coupling_types)
    df_test = filter_dataframes(df_test, coupling_types=coupling_types)
    
    if normalize_target:
        types_mean = [COUPLING_TYPE_MEAN[COUPLING_TYPE.index(t)] for t in df_train.type.values]
        types_std = [COUPLING_TYPE_STD[COUPLING_TYPE.index(t)] for t in df_train.type.values]
        df_train['scalar_coupling_constant'] = (df_train['scalar_coupling_constant'] - types_mean) / types_std
        df_test['scalar_coupling_constant'] = 0

    df_scalar_coupling = pd.concat([df_train, df_test], sort=False)

    ### scalar coupling contribution is not used here because we don't know the values for the test set
    #  
    #df_scalar_coupling_contribution = pd.read_csv(DATA_DIR + 'scalar_coupling_contributions.csv')
    #df_scalar_coupling = pd.merge(df_scalar_coupling, df_scalar_coupling_contribution,
    #                              how='left', on=['molecule_name', 'atom_index_0', 'atom_index_1' 'type'])

    #df = pd.DataFrame(df_structure.molecule_name.unique(), columns=['molecule_name'])
    #df['bond_indices'] = df.molecule_name.apply(lambda x: cis_trans_bond_indices(x, obConversion))
    #df['len_bond_indices'] = df.bond_indices.apply(lambda x:len(x))
    #df_structure = pd.merge(df_structure, df, how='left', on='molecule_name')

    gb_scalar_coupling = df_scalar_coupling.groupby('molecule_name')
    
    gb_structure = df_structure.groupby('molecule_name')

    return gb_structure, gb_scalar_coupling


def run_convert_to_graph(graph_dir='all_types', normalize_target=False, 
                         coupling_types=['1JHC', '2JHC', '3JHC', '1JHN', '2JHN', '3JHN', '2JHH', '3JHH']):
    graph_dir = get_path() + 'data/graphs/' + graph_dir
    os.makedirs(graph_dir, exist_ok=True)

    gb_structure, gb_scalar_coupling = load_csv(normalize_target, coupling_types=coupling_types)
    molecule_names = list(gb_scalar_coupling.groups.keys())
    molecule_names = np.sort(molecule_names)

    param = []

    for i, molecule_name in enumerate(molecule_names):
        graph_file = graph_dir + '/%s.pickle' % molecule_name
        p = molecule_name, gb_structure, gb_scalar_coupling, graph_file
        if i < 2000:
            do_one(p)
        else:
            param.append(p)

    pool = mp.Pool(processes=4)
    pool.map(do_one, param)


def do_one(p):
    molecule_name, gb_structure, gb_scalar_coupling, graph_file = p

    g = make_graph(molecule_name, gb_structure, gb_scalar_coupling)
    print(g.molecule_name, g.smiles)
    write_pickle_to_file(graph_file, g)


def make_graph(molecule_name, gb_structure, gb_scalar_coupling):
    # https://stackoverflow.com/questions/14734533/how-to-access-pandas-groupby-dataframe-by-key
    # ----
    df = gb_scalar_coupling.get_group(molecule_name)
    # ['id', 'molecule_name', 'atom_index_0', 'atom_index_1', 'type', 'scalar_coupling_constant', 'fc', 'sd', 'pso', 'dso', 
    # 'shortest_path_atoms', 'shortest_path_n_bonds', 'cosinus', 'dehidral'],

    coupling = Struct(
        id=df.id.values,
        #contribution=df[['fc', 'sd', 'pso', 'dso']].values,
        index=df[['atom_index_0', 'atom_index_1']].values,
        #type = np.array([ one_hot_encoding(t,COUPLING_TYPE) for t in df.type.values ], np.uint8)
        type=np.array([COUPLING_TYPE.index(t) for t in df.type.values], np.int32),
        value=df.scalar_coupling_constant.values,
    )

    # ----
    df = gb_structure.get_group(molecule_name)
    
    df = df.sort_values(['atom_index'], ascending=True)
    # ['molecule_name', 'atom_index', 'atom', 'x', 'y', 'z']
    a = df.atom.values.tolist()
    xyz = df[['x', 'y', 'z']].values
    mol = mol_from_axyz(a, xyz)

    yukawa_charges = df[['dist_C_0', 'dist_C_1', 'dist_C_2', 'dist_C_3', 'dist_C_4', 'dist_F_0',
                         'dist_F_1', 'dist_F_2', 'dist_F_3', 'dist_F_4', 'dist_H_0', 'dist_H_1',
                         'dist_H_2', 'dist_H_3', 'dist_H_4', 'dist_N_0', 'dist_N_1', 'dist_N_2',
                         'dist_N_3', 'dist_N_4', 'dist_O_0', 'dist_O_1', 'dist_O_2', 'dist_O_3',
                         'dist_O_4']].values

    # ---
    assert(a == [mol.GetAtomWithIdx(i).GetSymbol() for i in range(mol.GetNumAtoms())])

    # ---
    factory = ChemicalFeatures.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))
    feature = factory.GetFeaturesForMol(mol)

    # ** node **
    #[ a.GetSymbol() for a in mol.GetAtoms() ]

    num_atom = mol.GetNumAtoms()
    symbol = np.zeros((num_atom, len(SYMBOL)), np.uint8)  # category
    acceptor = np.zeros((num_atom, 1), np.uint8)
    donor = np.zeros((num_atom, 1), np.uint8)
    aromatic = np.zeros((num_atom, 1), np.uint8)
    hybridization = np.zeros((num_atom, len(HYBRIDIZATION)), np.uint8)
    num_h = np.zeros((num_atom, 1), np.float32)  # real
    atomic = np.zeros((num_atom, 1), np.float32)

    # these features seemed to help 
    radius = np.zeros((num_atom, 1), np.float32)  
    e = np.zeros((num_atom, 1), np.float32) 
    yukawa = np.zeros((num_atom, 25), np.float32) 
    # these features are new 
    mass = np.zeros((num_atom, 1), np.float32)
    degree = np.zeros((num_atom, 1), np.uint8)
    valence = np.zeros((num_atom, 1), np.uint8)
    in_ring = np.zeros((num_atom, 1), np.uint8)

    for t in range(0, len(feature)):
        if feature[t].GetFamily() == 'Donor':
            for i in feature[t].GetAtomIds():
                donor[i] = 1
        elif feature[t].GetFamily() == 'Acceptor':
            for i in feature[t].GetAtomIds():
                acceptor[i] = 1

    for i in range(num_atom):
        atom = mol.GetAtomWithIdx(i)
        symbol[i] = one_hot_encoding(atom.GetSymbol(), SYMBOL)
        aromatic[i] = atom.GetIsAromatic()
        hybridization[i] = one_hot_encoding(
            atom.GetHybridization(), HYBRIDIZATION)
        num_h[i] = atom.GetTotalNumHs(includeNeighbors=True)
        atomic[i] = atom.GetAtomicNum()
        # these features seemed to help 
        radius[i], e[i] = get_radius(atom.GetSymbol())
        yukawa[i] = yukawa_charges[i]
        ###### these features are new #######
        mass[i] = atom.GetMass()
        degree[i] = atom.GetDegree()
        valence[i] = atom.GetTotalValence()
        in_ring[i] = atom.IsInRing()
        ######################################

    # ** edge **
    num_edge = num_atom*num_atom - num_atom
    edge_index = np.zeros((num_edge, 2), np.uint8)
    bond_type = np.zeros((num_edge, len(BOND_TYPE)), np.uint8)  # category
    distance = np.zeros((num_edge, 3), np.float64) 
    angle = np.zeros((num_edge, 1), np.float32) 

    #valence_contrib = np.zeros((num_edge, 29), np.uint8) 
    conjugated = np.zeros((num_edge, 1), np.uint8) 

    norm_xyz = preprocessing.normalize(xyz, norm='l2')

    ij = 0
    for i in range(num_atom):
        for j in range(num_atom):
            if i == j:
                continue
            edge_index[ij] = [i, j]

            bond = mol.GetBondBetweenAtoms(i, j)
            if bond is not None:
                bond_type[ij] = one_hot_encoding(bond.GetBondType(), BOND_TYPE)
                conjugated[ij] = bond.GetIsConjugated()

                ### seemed not to help, probably doing the feature extraction wrong here
                #valence = []
                #for k in range(num_atom):
                #    atom = mol.GetAtomWithIdx(k)
                #    valence.append(int(bond.GetValenceContrib(atom)*2))
                #while len(valence) < 29:
                #    valence.append(0)
                #valence_contrib[ij] = valence
            
            distance[ij] = np.linalg.norm(xyz[i] - xyz[j], axis=0) 
            angle[ij] = (norm_xyz[i]*norm_xyz[j]).sum()
            ij += 1
    # -------------------

    graph = Struct(
        molecule_name=molecule_name,
        smiles=Chem.MolToSmiles(mol),
        axyz=[a, xyz],
        node=[symbol, acceptor, donor, aromatic, yukawa, degree,
              hybridization, num_h, atomic, radius, e, mass, in_ring],
        edge=[bond_type, distance, angle, conjugated], #valence_contrib
        edge_index=edge_index,
        coupling=coupling,
    )
    return graph


## xyz to mol #############################################################
# <todo> check for bug
# https://github.com/jensengroup/xyz2mol

def get_atom(atom):
    ATOM = [x.strip() for x in ['h ', 'he',
                                'li', 'be', 'b ', 'c ', 'n ', 'o ', 'f ', 'ne',
                                'na', 'mg', 'al', 'si', 'p ', 's ', 'cl', 'ar',
                                'k ', 'ca', 'sc', 'ti', 'v ', 'cr', 'mn', 'fe', 'co', 'ni', 'cu',
                                'zn', 'ga', 'ge', 'as', 'se', 'br', 'kr',
                                'rb', 'sr', 'y ', 'zr', 'nb', 'mo', 'tc', 'ru', 'rh', 'pd', 'ag',
                                'cd', 'in', 'sn', 'sb', 'te', 'i ', 'xe',
                                'cs', 'ba', 'la', 'ce', 'pr', 'nd', 'pm', 'sm', 'eu', 'gd', 'tb', 'dy',
                                'ho', 'er', 'tm', 'yb', 'lu', 'hf', 'ta', 'w ', 're', 'os', 'ir', 'pt',
                                'au', 'hg', 'tl', 'pb', 'bi', 'po', 'at', 'rn',
                                'fr', 'ra', 'ac', 'th', 'pa', 'u ', 'np', 'pu']]
    atom = atom.lower()
    return ATOM.index(atom) + 1


def get_radius(atom):    
    return R[atom], E[atom]


def getUA(maxValence_list, valence_list):
    UA = []
    DU = []
    for i, (maxValence, valence) in enumerate(zip(maxValence_list, valence_list)):
        if maxValence - valence > 0:
            UA.append(i)
            DU.append(maxValence - valence)
    return UA, DU


def get_BO(AC, UA, DU, valences, UA_pairs, quick):
    BO = AC.copy()
    DU_save = []

    while DU_save != DU:
        for i, j in UA_pairs:
            BO[i, j] += 1
            BO[j, i] += 1

        BO_valence = list(BO.sum(axis=1))
        DU_save = copy.copy(DU)
        UA, DU = getUA(valences, BO_valence)
        UA_pairs = get_UA_pairs(UA, AC, quick)[0]

    return BO


def valences_not_too_large(BO, valences):
    number_of_bonds_list = BO.sum(axis=1)
    for valence, number_of_bonds in zip(valences, number_of_bonds_list):
        if number_of_bonds > valence:
            return False

    return True


def BO_is_OK(BO, AC, charge, DU, atomic_valence_electrons, atomicNumList, charged_fragments):
    Q = 0  # total charge
    q_list = []
    if charged_fragments:
        BO_valences = list(BO.sum(axis=1))
        for i, atom in enumerate(atomicNumList):
            q = get_atomic_charge(
                atom, atomic_valence_electrons[atom], BO_valences[i])
            Q += q
            if atom == 6:
                number_of_single_bonds_to_C = list(BO[i, :]).count(1)
                if number_of_single_bonds_to_C == 2 and BO_valences[i] == 2:
                    Q += 1
                    q = 2
                if number_of_single_bonds_to_C == 3 and Q + 1 < charge:
                    Q += 2
                    q = 1

            if q != 0:
                q_list.append(q)

    if (BO-AC).sum() == sum(DU) and charge == Q and len(q_list) <= abs(charge):
        return True
    else:
        return False


def get_atomic_charge(atom, atomic_valence_electrons, BO_valence):
    if atom == 1:
        charge = 1 - BO_valence
    elif atom == 5:
        charge = 3 - BO_valence
    elif atom == 15 and BO_valence == 5:
        charge = 0
    elif atom == 16 and BO_valence == 6:
        charge = 0
    else:
        charge = atomic_valence_electrons - 8 + BO_valence

    return charge


def clean_charges(mol):
    # this hack should not be needed any more but is kept just in case

    rxn_smarts = ['[N+:1]=[*:2]-[C-:3]>>[N+0:1]-[*:2]=[C-0:3]',
                  '[N+:1]=[*:2]-[O-:3]>>[N+0:1]-[*:2]=[O-0:3]',
                  '[N+:1]=[*:2]-[*:3]=[*:4]-[O-:5]>>[N+0:1]-[*:2]=[*:3]-[*:4]=[O-0:5]',
                  '[#8:1]=[#6:2]([!-:6])[*:3]=[*:4][#6-:5]>>[*-:1][*:2]([*:6])=[*:3][*:4]=[*+0:5]',
                  '[O:1]=[c:2][c-:3]>>[*-:1][*:2][*+0:3]',
                  '[O:1]=[C:2][C-:3]>>[*-:1][*:2]=[*+0:3]']

    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)

    for i, fragment in enumerate(fragments):
        for smarts in rxn_smarts:
            patt = Chem.MolFromSmarts(smarts.split(">>")[0])
            while fragment.HasSubstructMatch(patt):
                rxn = AllChem.ReactionFromSmarts(smarts)
                ps = rxn.RunReactants((fragment,))
                fragment = ps[0][0]
        if i == 0:
            mol = fragment
        else:
            mol = Chem.CombineMols(mol, fragment)

    return mol


def BO2mol(mol, BO_matrix, atomicNumList, atomic_valence_electrons, mol_charge, charged_fragments):
    # based on code written by Paolo Toscani
    l = len(BO_matrix)
    l2 = len(atomicNumList)
    BO_valences = list(BO_matrix.sum(axis=1))

    if (l != l2):
        raise RuntimeError('sizes of adjMat ({0:d}) and atomicNumList '
                           '{1:d} differ'.format(l, l2))

    rwMol = Chem.RWMol(mol)

    bondTypeDict = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE
    }

    for i in range(l):
        for j in range(i + 1, l):
            bo = int(round(BO_matrix[i, j]))
            if (bo == 0):
                continue
            bt = bondTypeDict.get(bo, Chem.BondType.SINGLE)
            rwMol.AddBond(i, j, bt)
    mol = rwMol.GetMol()

    if charged_fragments:
        mol = set_atomic_charges(
            mol, atomicNumList, atomic_valence_electrons, BO_valences, BO_matrix, mol_charge)
    else:
        mol = set_atomic_radicals(
            mol, atomicNumList, atomic_valence_electrons, BO_valences)

    return mol


def set_atomic_charges(mol, atomicNumList, atomic_valence_electrons, BO_valences, BO_matrix, mol_charge):
    q = 0
    for i, atom in enumerate(atomicNumList):
        a = mol.GetAtomWithIdx(i)
        charge = get_atomic_charge(
            atom, atomic_valence_electrons[atom], BO_valences[i])
        q += charge
        if atom == 6:
            number_of_single_bonds_to_C = list(BO_matrix[i, :]).count(1)
            if number_of_single_bonds_to_C == 2 and BO_valences[i] == 2:
                q += 1
                charge = 0
            if number_of_single_bonds_to_C == 3 and q + 1 < mol_charge:
                q += 2
                charge = 1

        if (abs(charge) > 0):
            a.SetFormalCharge(int(charge))

    # shouldn't be needed anymore bit is kept just in case
    #mol = clean_charges(mol)

    return mol


def set_atomic_radicals(mol, atomicNumList, atomic_valence_electrons, BO_valences):
    # The number of radical electrons = absolute atomic charge
    for i, atom in enumerate(atomicNumList):
        a = mol.GetAtomWithIdx(i)
        charge = get_atomic_charge(
            atom, atomic_valence_electrons[atom], BO_valences[i])

        if (abs(charge) > 0):
            a.SetNumRadicalElectrons(abs(int(charge)))

    return mol


def get_bonds(UA, AC):
    bonds = []

    for k, i in enumerate(UA):
        for j in UA[k+1:]:
            if AC[i, j] == 1:
                bonds.append(tuple(sorted([i, j])))

    return bonds


def get_UA_pairs(UA, AC, quick):
    bonds = get_bonds(UA, AC)
    if len(bonds) == 0:
        return [()]

    if quick:
        G = nx.Graph()
        G.add_edges_from(bonds)
        UA_pairs = [list(nx.max_weight_matching(G))]
        return UA_pairs

    max_atoms_in_combo = 0
    UA_pairs = [()]
    for combo in list(itertools.combinations(bonds, int(len(UA)/2))):
        flat_list = [item for sublist in combo for item in sublist]
        atoms_in_combo = len(set(flat_list))
        if atoms_in_combo > max_atoms_in_combo:
            max_atoms_in_combo = atoms_in_combo
            UA_pairs = [combo]
            # if quick and max_atoms_in_combo == 2*int(len(UA)/2):
            #    return UA_pairs
        elif atoms_in_combo == max_atoms_in_combo:
            UA_pairs.append(combo)

    return UA_pairs


def AC2BO(AC, atomicNumList, charge, charged_fragments, quick):
    # TODO
    atomic_valence = defaultdict(list)
    atomic_valence[1] = [1]
    atomic_valence[6] = [4]
    atomic_valence[7] = [4, 3]
    atomic_valence[8] = [2, 1]
    atomic_valence[9] = [1]
    atomic_valence[14] = [4]
    atomic_valence[15] = [5, 4, 3]
    atomic_valence[16] = [6, 4, 2]
    atomic_valence[17] = [1]
    atomic_valence[32] = [4]
    atomic_valence[35] = [1]
    atomic_valence[53] = [1]

    atomic_valence_electrons = {}
    atomic_valence_electrons[1] = 1
    atomic_valence_electrons[6] = 4
    atomic_valence_electrons[7] = 5
    atomic_valence_electrons[8] = 6
    atomic_valence_electrons[9] = 7
    atomic_valence_electrons[14] = 4
    atomic_valence_electrons[15] = 5
    atomic_valence_electrons[16] = 6
    atomic_valence_electrons[17] = 7
    atomic_valence_electrons[32] = 4
    atomic_valence_electrons[35] = 7
    atomic_valence_electrons[53] = 7

    # make a list of valences, e.g. for CO: [[4],[2,1]]
    valences_list_of_lists = []
    for atomicNum in atomicNumList:
        valences_list_of_lists.append(atomic_valence[atomicNum])

    # convert [[4],[2,1]] to [[4,2],[4,1]]
    valences_list = list(itertools.product(*valences_list_of_lists))

    best_BO = AC.copy()

    # implemenation of algorithm shown in Figure 2
    # UA: unsaturated atoms
    # DU: degree of unsaturation (u matrix in Figure)
    # best_BO: Bcurr in Figure
    #

    for valences in valences_list:
        AC_valence = list(AC.sum(axis=1))
        UA, DU_from_AC = getUA(valences, AC_valence)

        if len(UA) == 0 and BO_is_OK(AC, AC, charge, DU_from_AC, atomic_valence_electrons, atomicNumList, charged_fragments):
            return AC, atomic_valence_electrons

        UA_pairs_list = get_UA_pairs(UA, AC, quick)
        for UA_pairs in UA_pairs_list:
            BO = get_BO(AC, UA, DU_from_AC, valences, UA_pairs, quick)
            if BO_is_OK(BO, AC, charge, DU_from_AC, atomic_valence_electrons, atomicNumList, charged_fragments):
                return BO, atomic_valence_electrons

            elif BO.sum() >= best_BO.sum() and valences_not_too_large(BO, valences):
                best_BO = BO.copy()

    return best_BO, atomic_valence_electrons


def AC2mol(mol, AC, atomicNumList, charge, charged_fragments, quick):
    # convert AC matrix to bond order (BO) matrix
    BO, atomic_valence_electrons = AC2BO(
        AC, atomicNumList, charge, charged_fragments, quick)

    # add BO connectivity and charge info to mol object
    mol = BO2mol(mol, BO, atomicNumList, atomic_valence_electrons,
                 charge, charged_fragments)

    return mol


def get_proto_mol(atomicNumList):
    mol = Chem.MolFromSmarts("[#"+str(atomicNumList[0])+"]")
    rwMol = Chem.RWMol(mol)
    for i in range(1, len(atomicNumList)):
        a = Chem.Atom(atomicNumList[i])
        rwMol.AddAtom(a)

    mol = rwMol.GetMol()

    return mol


def get_atomicNumList(atomic_symbols):
    atomicNumList = []
    for symbol in atomic_symbols:
        atomicNumList.append(get_atom(symbol))
    return atomicNumList


def xyz2AC(atomicNumList, xyz):

    mol = get_proto_mol(atomicNumList)

    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, (xyz[i][0], xyz[i][1], xyz[i][2]))
    mol.AddConformer(conf)

    dMat = Chem.Get3DDistanceMatrix(mol)
    pt = Chem.GetPeriodicTable()

    num_atoms = len(atomicNumList)
    AC = np.zeros((num_atoms, num_atoms)).astype(int)

    for i in range(num_atoms):
        a_i = mol.GetAtomWithIdx(i)
        Rcov_i = pt.GetRcovalent(a_i.GetAtomicNum())*1.30
        for j in range(i+1, num_atoms):
            a_j = mol.GetAtomWithIdx(j)
            Rcov_j = pt.GetRcovalent(a_j.GetAtomicNum())*1.30
            if dMat[i, j] <= Rcov_i + Rcov_j:
                AC[i, j] = 1
                AC[j, i] = 1

    return AC, mol


def read_xyz_file(filename):
    atomic_symbols = []
    xyz_coordinates = []

    with open(filename, "r") as file:
        for line_number, line in enumerate(file):
            if line_number == 0:
                num_atoms = int(line)
            elif line_number == 1:
                if "charge=" in line:
                    charge = int(line.split("=")[1])
                else:
                    charge = 0
            else:
                atomic_symbol, x, y, z = line.split()
                atomic_symbols.append(atomic_symbol)
                xyz_coordinates.append([float(x), float(y), float(z)])

    atomicNumList = get_atomicNumList(atomic_symbols)
    return atomicNumList, xyz_coordinates, charge

# -----
# https://www.kaggle.com/sunhwan/using-rdkit-for-atomic-feature-and-visualization


def chiral_stereo_check(mol):
    # avoid sanitization error e.g., dsgdb9nsd_037900.xyz
    Chem.SanitizeMol(mol, SanitizeFlags.SANITIZE_ALL -
                     SanitizeFlags.SANITIZE_PROPERTIES)
    Chem.DetectBondStereochemistry(mol, -1)

    # ignore stereochemistry for now
    Chem.AssignStereochemistry(mol, flagPossibleStereoCenters=True, force=True)
    Chem.AssignAtomChiralTagsFromStructure(mol, -1)
    return mol


def xyz2mol(atomicNumList, xyz_coordinates, charge, charged_fragments, quick):
    AC, mol = xyz2AC(atomicNumList, xyz_coordinates)
    new_mol = AC2mol(mol, AC, atomicNumList, charge, charged_fragments, quick)
    new_mol = chiral_stereo_check(new_mol)
    return new_mol


def MolFromXYZ(filename):
    charged_fragments = True
    quick = True
    atomicNumList, xyz_coordinates, charge = (filename)
    mol = xyz2mol(atomicNumList, xyz_coordinates,
                  charge, charged_fragments, quick)
    return mol


## champs dataset #############################################################
'''
dsgdb9nsd_000001.xyz

5

C -0.0126981359 1.0858041578 0.0080009958
H 0.0021504160 -0.0060313176 0.0019761204
H 1.0117308433 1.4637511618 0.0002765748
H -0.5408150690 1.4475266138 -0.8766437152
H -0.5238136345 1.4379326443 0.9063972942

'''

def read_champs_xyz(xyz_file):
    line = read_list_from_file(xyz_file, comment=None)
    num_atom = int(line[0])
    xyz = []
    symbol = []
    for n in range(num_atom):
        l = line[1+n]
        l = l.replace('\t', ' ').replace('  ', ' ')
        l = l.split(' ')
        symbol.append(l[0])
        xyz.append([float(l[1]), float(l[2]), float(l[3]), ])

    return symbol, xyz


def mol_from_axyz(symbol, xyz):
    charged_fragments = True
    quick = True
    charge = 0
    atom_no = get_atomicNumList(symbol)
    mol = xyz2mol(atom_no, xyz, charge, charged_fragments, quick)
    return mol


def run_make_split(num_valid, name='by_mol', graphs='all_types', coupling_types=None):
    split_dir = get_path() + 'data/split/'
    csv_file = get_data_path() + 'train.csv'
    os.makedirs(split_dir, exist_ok=True)

    df = pd.read_csv(csv_file)
    
    if coupling_types is not None:
        df = df.loc[df['type'].isin(coupling_types)]

    molecule_names = df.molecule_name.unique()
    molecule_names = np.sort(molecule_names)

    debug_split = molecule_names[:1000]
    np.save(split_dir + 'debug_split_by_mol.%d.npy' %
            len(debug_split), debug_split)

    np.random.shuffle(molecule_names)
    num_all = len(molecule_names)
    num_valid = num_valid
    num_train = num_all - num_valid
    print(num_train, num_valid)
    train_split = molecule_names[num_valid:]
    valid_split = molecule_names[:num_valid]

    np.save(split_dir + 'train_split_%s.%d.npy' % (name, num_train), train_split)
    np.save(split_dir + 'valid_split_%s.%d.npy' % (name, num_valid), valid_split)


# check #######################################################################
def run_check_0():
    xyz_dir = get_path() + 'data/structures/xyz'
    name = [
        'dsgdb9nsd_000001',
        'dsgdb9nsd_000002',
        'dsgdb9nsd_000005',
        'dsgdb9nsd_000007',
        'dsgdb9nsd_037490',
        'dsgdb9nsd_037493',
        'dsgdb9nsd_037494',
    ]
    for n in name:
        xyz_file = xyz_dir + '/%s.xyz' % n

        symbol, xyz = read_champs_xyz(xyz_file)
        mol = mol_from_axyz(symbol, xyz)
        
        smiles = Chem.MolToSmiles(mol)
        print(n, smiles)

        image = np.array(Chem.Draw.MolToImage(mol, size=(128, 128)))
        image_show('', image)
        cv2.waitKey(0)


def run_check_0a():

    gb_structure, gb_scalar_coupling = load_csv()

    molecule_name = 'dsgdb9nsd_000001'
    graph = make_graph(molecule_name, gb_structure, gb_scalar_coupling)

    print(graph)
    print('graph.molecule_name:', graph.molecule_name)
    print('graph.smiles:', graph.smiles)
    print('graph.node:', np.concatenate(graph.node, -1).shape)
    print('graph.edge:', np.concatenate(graph.edge, -1).shape)
    print('graph.edge_index:', graph.edge_index.shape)
    print('-----')
    print('graph.coupling.index:', graph.coupling.index.shape)
    print('graph.coupling.type:', graph.coupling.type.shape)
    print('graph.coupling.value:', graph.coupling.value.shape)
    #print('graph.coupling.contribution:', graph.coupling.contribution.shape)
    print('graph.coupling.id:', graph.coupling.id)


# main #################################################################
if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))
    #1JHC, 2JHC, 3JHC, 1JHN, 2JHN, 3JHN, 2JHH, 3JHH
    coupling_types = ['1JHC', '2JHC', '3JHC', '1JHN', '2JHN', '3JHN', '2JHH', '3JHH']
    run_convert_to_graph(graph_dir='all_types_selected_features', normalize_target=False, coupling_types=coupling_types)
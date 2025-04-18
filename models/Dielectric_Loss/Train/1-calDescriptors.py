from multiprocessing import freeze_support
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from mordred import Calculator, descriptors
import numpy as np
np.float=float
np.int=int
# Generates all descriptors of the structure
def cal_descriptors_by_mordred(mol_list):
    freeze_support()
    # Instantiate an object——All descriptors calculator
    calculator = Calculator(descriptors)
    mordred_pd = calculator.pandas(mol_list)
    return mordred_pd


def main():
    # Reading data
    data_all_pd = pd.read_csv('Test_Structure_Data.csv', usecols=['RepeatUnitSMILES', 'PropertyValue', 'TestFreq','TestTemp'])
    data_all = data_all_pd.values
    # Get all SMILES
    SMILES_list = data_all[:, 0].reshape(1, -1).tolist()[0]
    print(SMILES_list)

    # Convert to a mol object
    mols_list = [Chem.MolFromSmiles(smiles) for smiles in SMILES_list]

    # AddHs
    molsAddHs_list = [AllChem.AddHs(mol) for mol in mols_list]

    # Calculate descriptors and save them as pandas Dataframe
    descriptors_pd = cal_descriptors_by_mordred(molsAddHs_list)

    # Calculate descriptors and save them as pandas Dataframe
    data_pd_out = pd.concat([data_all_pd, descriptors_pd], axis=1)

    # save file
    data_pd_out.to_csv('Tset_all_descriptors.csv', index=False)


if __name__ == '__main__':
    main()

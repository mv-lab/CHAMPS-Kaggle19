import os
import numpy as np
import pandas as pd
from common import get_path, get_data_path


def run_ensamble():
    ensemble_file = get_path() + 'data/submission/ensamble/mpnn_ensemble_all_good.csv'

    df_test = pd.read_csv(get_data_path() + 'test.csv')

    df_test = df_test.sort_values(by=['id'])

    test_id = df_test.id.values
    test_type = df_test.type.values

    num_test = len(test_id)
    coupling_count = np.zeros(num_test)
    coupling_value = np.zeros(num_test)

    # best local cv:
    # 0.000020  228.5*  85.7 |  -1.130, -1.981, -1.651, -1.140, -2.251, -2.310, -2.133, -1.777 | -1.694  0.18 -1.80 | -2.058 |  3 hr 15 min
    # 0.00008   237.5*  47.5 |  -1.178, -1.935, -1.590, +0.000, +0.000, +0.000, +0.000, +0.000 | -1.596  0.22 -1.57 | -1.559 |  0 hr 36 min

    subs = [
        (['2JHC', '3JHC', '1JHN', '2JHN', '3JHN', '2JHH', '3JHH'],
         get_path() + 'data/submission/all_types/submit/sagpooling_larger_mpnn-175.csv'),
        (['1JHC', '2JHC', '3JHC'],
         get_path() + 'data/submission/all_JHC/submit/submit-00237500_model-larger.csv'),
        (['1JHC', '2JHC', '3JHC', '1JHN', '2JHN', '3JHN', '2JHH', '3JHH'],
         get_path() + 'data/submission/all_types_yukawa_radius_set2set/submit/set2set_highbs_mpnn_larger.csv')
    ]

    for valid_type, f in subs:
        df = pd.read_csv(f)
        df = df.sort_values(by=['id'])
        for t in valid_type:
            index = np.where(test_type == t)[0]
            coupling_value[index] += df.scalar_coupling_constant.values[index]
            coupling_count[index] += 1

    coupling_value = coupling_value/coupling_count

    df = pd.DataFrame(list(zip(test_id, coupling_value)),
                      columns=['id', 'scalar_coupling_constant'])
    df.to_csv(ensemble_file, index=False)


# main #################################################################
if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

    run_ensamble()

    #asdf1 = pd.read_csv(get_path() + 'data/submission/ensamble/mpnn-ensemble-sagpool.csv')
    #asdf2 = pd.read_csv(get_path() + 'data/submission/zzz/submit/submit-00325000_model-larger.csv')

    # print(asdf1.describe())
    # print(asdf2.describe())
    # print(asdf1.scalar_coupling_constant.mean()-asdf2.scalar_coupling_constant.mean())

    print('\nsuccess!')

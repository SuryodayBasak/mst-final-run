import pandas as pd
import numpy as np

def show_res(df, op, sw):
    
    if op == 'xx' and sw == 'knn':
        m_name = 'KNN'

    elif op == 'xx' and sw == 'dknn':
        m_name = 'D-KNN'

    elif op == 'ga' and sw == 'knn':
        m_name = 'GA-KNN'

    elif op == 'ga' and sw == 'dknn':
        m_name = 'GA-D-KNN'

    elif op == 'gbest-pso' and sw == 'knn':
        m_name = 'GPSO-KNN'

    elif op == 'gbest-pso' and sw == 'dknn':
        m_name = 'GPSO-D-KNN'

    elif op == 'lbest-pso' and sw == 'knn':
        m_name = 'LPSO-KNN'

    elif op == 'lbest-pso' and sw == 'dknn':
        m_name = 'LPSO-D-KNN'

    res_list = []
    mse_list = [3, 5, 7, 9, 11, 13, 15]

    df_subset = df.loc[df['optimizer'] == op]
    df_subset = df_subset.loc[df_subset['sample_weights'] == sw]

    for i in mse_list:
        k_i = df_subset.loc[df_subset['k'] == i]
        col = k_i.loc[:, 'mse']
        res_list.append(sum(col)/len(k_i))

    print('&', m_name,\
          '&',round(res_list[0], 2),\
          '&',round(res_list[1], 2),\
          '&',round(res_list[2], 2),\
          '&',round(res_list[3], 2),\
          '&',round(res_list[4], 2),\
          '&',round(res_list[5], 2),\
          '&',round(res_list[6], 2),\
          '\\\\')

col_names = ['optimizer', 'sample_weights', 'k', 'mse']
df = pd.read_csv('results_dump.csv',
                  header=None,
                  names=col_names)

# No optimizer, knn.
show_res(df, 'xx', 'knn')

# No optimizer, dknn.
show_res(df, 'xx', 'dknn')

# GA, KNN.
show_res(df, 'ga', 'knn')

# GA, DKNN.
show_res(df, 'ga', 'dknn')

# GBest PSO, KNN.
show_res(df, 'gbest-pso', 'knn')

# GBest PSO, DKNN.
show_res(df, 'gbest-pso', 'dknn')

# LBest PSO, KNN.
show_res(df, 'lbest-pso', 'knn')

# LBest PSO, DKNN.
show_res(df, 'lbest-pso', 'dknn')

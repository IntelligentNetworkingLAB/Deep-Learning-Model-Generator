import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def find_nearest_neighbor(df_data, find_mem_from, find_mem_basedon, num_neighbors, day, window_size, sequence_by):
    day_range = day - np.flip(np.arange(window_size), 0)
    # To find the similar neigbor for training
    potential_member = df_data[find_mem_from].unique()
    dict_tmp = {k: v for v, k in enumerate(potential_member)}
    dict_tmp = {y: x for x, y in dict_tmp.items()}
    X = df_data[df_data[sequence_by].isin(day_range)].reset_index().pivot(index=find_mem_from, columns=sequence_by,
                                                                          values=find_mem_basedon)[day_range]
    X_index = X.index
    X = X.values

    nbrs = NearestNeighbors(n_neighbors=num_neighbors, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)
    tmp_df = pd.DataFrame(indices)
    tmp_df = tmp_df.replace(dict_tmp)
    X = tmp_df.values

    return X


def get_train_data(df_data, find_mem_from, find_mem_basedon, sequence_by, window_size, num_neighbors, features_cols,
                   target_label):
    x_train = []
    y_train = pd.DataFrame()
    for day in df_data.dayofyear.unique()[window_size - 1:-1]:
        n_neighbor = find_nearest_neighbor(df_data, find_mem_from, find_mem_basedon, num_neighbors, day, window_size,
                                           sequence_by)
        #         print(n_neighbor)
        day_range = day - np.flip(np.arange(window_size), 0)
        #         print(day_range)
        df_tmp = df_data[df_data[sequence_by].isin(day_range)]
        #         print(df_tmp)

        for i in range(len(df_tmp[find_mem_from].unique())):
            x_tmp = df_tmp[df_tmp[find_mem_from].isin(n_neighbor[i])][features_cols]
            #             print(x_tmp)
            for r in range(int(len(x_tmp) / num_neighbors)):
                for k in range(num_neighbors):
                    if k == 0:
                        row_num = r
                        x_train.append(x_tmp.iloc[row_num].values)
                    else:
                        row_num += window_size
                        x_train.append(x_tmp.iloc[row_num].values)

        y_tmp = df_tmp[df_tmp[sequence_by].isin([day])][target_label]
        y_train = y_train.append(y_tmp, ignore_index=True)

    x_train = np.reshape(x_train, (
    len(df_data.dayofyear.unique()[window_size - 1:-1]) * len(df_data[find_mem_from].unique()), window_size,
    num_neighbors * len(features_cols)))
    y_train = y_train.values

    return x_train, y_train   
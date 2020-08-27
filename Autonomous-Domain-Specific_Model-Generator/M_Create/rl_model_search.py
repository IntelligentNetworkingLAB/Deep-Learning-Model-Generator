import pandas as pd
import numpy as np
import itertools
import re

def gen_com_model(model_list, max_layers, layer_step_size):
    model_info = dict()
    combined_model = 0
    min_layers = 0
    for i in model_list:
        each_archi = dict()
        if i == "CLSTM":
            combined_model = 3
            min_layers = 3
        else:
            combined_model = 2
            min_layers = 2

        layers_list = np.arange(min_layers, max_layers + 1, layer_step_size)

        for j in layers_list:
            layers_config = np.arange(j)
            if i != "CLSTM":
                layers_config = [config for config in itertools.product(layers_config, repeat=combined_model) if
                                 (sum(config) == j) & (config[0] != 0) & (config[1] != 0)]
                each_archi[j] = layers_config
            else:
                layers_config = [config for config in itertools.product(layers_config, repeat=combined_model) if
                                 (sum(config) == j) &
                                 (config[0] != 0) & (config[1] != 0) & (config[2] != 0)]
                each_archi[j] = layers_config
        model_info[i] = each_archi
    return model_info

def gen_searchspace(model_type,config_choice,min_cells,max_cells,cell_step_size):
    config_tmp = []
    df_tmp = pd.DataFrame()
    counter=0
    cell_list=np.arange(min_cells,max_cells+1,cell_step_size)
#     print(config_choice)
    for each_archi_layer in config_choice:
#         print('each_archi',each_archi_layer)
        cell_tmp = []
        for i in range(each_archi_layer):
            cell_tmp.append(np.random.choice(cell_list))
#             print(cell_tmp)
#         print(str(list(filter(None, cell_tmp))))
        config_tmp.append(str(list(filter(None, cell_tmp))))
        counter +=1
    df_tmp[counter]=config_tmp
    df_tmp = df_tmp.transpose()
#     print(df_tmp.transpose())
#     print(config_tmp)
#     df_tmp = pd.DataFrame(str(config_tmp))
#     df_tmp = df_tmp.transpose()
    df_tmp['model_type'] = model_type
#     print('Gen_config_check', [config_choice])
    df_tmp['config_type'] = str(config_choice)
    df_tmp['layer_type'] = sum(config_choice)
#     df_tmp2 = df_tmp
#     print('search_space',df_tmp)
    return df_tmp

def rand_explore_space(model_info, min_cells, max_cells, cell_step_size):
    #     print(model_info)

    mod_type_arr = []
    layer_arr = []
    config_arr = []

    if len(model_info) != 0:
        df_tmp = pd.DataFrame()
        model_type = list(model_info.keys())
        model_type = np.random.choice(model_type)

        layer_type = list(model_info[model_type].keys())
        layer_type = np.random.choice(layer_type)

        config_index = np.arange(len(list(model_info[model_type][layer_type])))
        config_index = np.random.choice(config_index)
        config_type = model_info[model_type][layer_type][config_index]
        model_info[model_type][layer_type].pop(config_index)

        if len(model_info[model_type][layer_type]) == 0:
            model_info[model_type].pop(layer_type)

        if len(model_info[model_type]) == 0:
            model_info.pop(model_type)

    return model_info, model_type, layer_type, config_type


# Exploration
def exploration(model_info, model_tab, layer_tab, config_tab,min_cells, max_cells, cell_step_size):
    model_info, model_dir, layer_dir, config_name = rand_explore_space(model_info, min_cells, max_cells, cell_step_size)
    to_train_df = gen_searchspace(model_dir, config_name, min_cells, max_cells, cell_step_size)

    return to_train_df, model_info, model_tab, layer_tab, config_tab


# exploitation
def exploitation(model_tab, layer_tab, config_tab,min_cells, max_cells, cell_step_size):
    model_type = model_tab[model_tab['q'] == model_tab['q'].min()]
    model_type = model_type['model_type'].values[0]

    layer_type = layer_tab[(layer_tab['model_type'] == model_type)]
    layer_type = layer_type[layer_type['q'] == layer_type['q'].min()]
    layer_type = layer_type['layer_type'].values[0]

    config_type = config_tab[(config_tab['model_type'] == model_type) & (config_tab['layer_type'] == layer_type)]
    config_type = config_type[config_tab['q'] == config_type['q'].min()]
    config_type = config_type['config_type'].values[0]
    config_re = re.compile(r'\d+(?:,\d+)?')
    config_type = tuple(list(map(int, config_re.findall(config_type))))

    to_train_df = gen_searchspace(model_type, config_type, min_cells, max_cells, cell_step_size)

    return to_train_df, model_tab, layer_tab, config_tab

def q_fun(alpha,r,gamma,model_next_min,model_old):
    q_value = alpha*(r+(gamma*(model_next_min)-model_old))
    return q_value


def up_model_q(r, model_tab, model_tab_log, model_type, layer_tab, layer_type, alpha, gamma):
    model_old = 0
    model_next_min = 0

    # model q update
    if len(model_tab) == 0:  # no log info
        q_value = model_old + q_fun(alpha, r, gamma, model_next_min, model_old)
        model_tab = pd.DataFrame({'model_type': [model_type], 'q': [q_value]})
        model_tab_log = pd.DataFrame({'model_type': [model_type], 'q': [q_value]})
    else:
        if len(model_tab[model_tab['model_type'] == model_type]) == 0:
            q_value = model_old + q_fun(alpha, r, gamma, model_next_min, model_old)
            df_tmp = pd.DataFrame({'model_type': [model_type], 'q': [q_value]})
            model_tab = model_tab.append(df_tmp, ignore_index=True)
            model_tab_log = model_tab_log.append(df_tmp, ignore_index=True)
        else:
            model_old = model_tab[(model_tab['model_type'] == model_type)]['q'].values[0]
            if len(layer_tab[(layer_tab['model_type'] == model_type) & (layer_tab['layer_type'] == layer_type)]) != 0:
                # get min q_value
                df_tmp = layer_tab[(layer_tab['model_type'] == model_type) & (layer_tab['layer_type'] == layer_type)]
                df_tmp = df_tmp[df_tmp['q'] == df_tmp['q'].min()]
                model_next_min = df_tmp.q.values[0]

            q_value = model_old + q_fun(alpha, r, gamma, model_next_min, model_old)
            model_tab.loc[(model_tab['model_type'] == model_type), 'q'] = q_value
            model_tab_log.loc[(model_tab_log['model_type'] == model_type), 'q'] = q_value
    return model_tab, model_tab_log


def up_layer_q(r, layer_tab, layer_tab_log, model_type, layer_type, config_tab, config_type, alpha, gamma):
    layer_old = 0
    layer_next_min = 0

    # layer q update
    if len(layer_tab) == 0:  # no log info
        q_value = layer_old + q_fun(alpha, r, gamma, layer_next_min, layer_old)
        layer_tab = pd.DataFrame({'model_type': [model_type], 'layer_type': [layer_type], 'q': [r]})
        layer_tab_log = pd.DataFrame({'model_type': [model_type], 'layer_type': [layer_type], 'q': [r]})
    else:
        if len(layer_tab[(layer_tab['model_type'] == model_type) & (layer_tab['layer_type'] == layer_type)]) == 0:
            q_value = layer_old + q_fun(alpha, r, gamma, layer_next_min, layer_old)
            df_tmp = pd.DataFrame({'model_type': [model_type], 'layer_type': [layer_type], 'q': [q_value]})
            layer_tab = layer_tab.append(df_tmp, ignore_index=True)
            layer_tab_log = layer_tab_log.append(df_tmp, ignore_index=True)
        else:
            layer_old = layer_tab[(layer_tab['model_type'] == model_type) &
                                  (layer_tab['layer_type'] == layer_type)]['q'].values[0]
            if len(config_tab[(config_tab['model_type'] == model_type) &
                    (config_tab['layer_type'] == layer_type) &
                    (str(config_tab['config_type']) == str(config_type))]) != 0:
                # get min q_value
                df_tmp = config_tab[(config_tab['model_type'] == model_type) &
                                    (config_tab['layer_type'] == layer_type) &
                                    (str(config_tab['config_type']) == str(config_type))]
                df_tmp = df_tmp[df_tmp['q'] == df_tmp['q'].min()]
                model_next_min = df_tmp.q.values[0]

            q_value = layer_old + q_fun(alpha, r, gamma, layer_next_min, layer_old)
            layer_tab.loc[(layer_tab['model_type'] == model_type) &
                          (layer_tab['layer_type'] == layer_type), 'q'] = q_value
            layer_tab_log.loc[(layer_tab_log['model_type'] == model_type) &
                              (layer_tab_log['layer_type'] == layer_type), 'q'] = q_value
    return layer_tab, layer_tab_log

def up_config_q(r,config_tab,config_tab_log,model_type,layer_type,config_type,alpha,gamma):
    config_old = 0
    config_next_min = 0
    #config q update
    if len(config_tab) == 0:#no log info
        q_value = config_old + q_fun(alpha,r,gamma,config_next_min,config_old)
        config_tab = pd.DataFrame({'model_type':[model_type],'layer_type':[layer_type],'config_type':[config_type],'q':[q_value]})
        config_tab_log = pd.DataFrame({'model_type':[model_type],'layer_type':[layer_type],'config_type':[config_type],'q':[q_value]})
    else:
        if len(config_tab[(config_tab['model_type']==model_type)&
                          (config_tab['layer_type']==layer_type)&
                          (str(config_tab['config_type'])==str(config_type))])==0:
            q_value = config_old + q_fun(alpha,r,gamma,config_next_min,config_old)
            df_tmp = pd.DataFrame({'model_type':[model_type],'layer_type':[layer_type],'config_type':[config_type],'q':[q_value]})
            config_tab= config_tab.append(df_tmp,ignore_index=True)
            config_tab_log= config_tab_log.append(df_tmp,ignore_index=True)
        else:
            config_old = config_tab[(config_tab['model_type']==model_type)&
                                  (config_tab['layer_type']==layer_type)&
                                   (str(config_tab['config_type'])==str(config_type))]['q'].values[0]
            q_value = config_old + q_fun(alpha,r,gamma,config_next_min,config_old)
            config_tab.loc[(config_tab['model_type']==model_type)&
                          (config_tab['layer_type']==layer_type)&
                          (str(config_tab['config_type'])==str(config_type)),'q'] = q_value
            config_tab_log.loc[(config_tab_log['model_type']==model_type)&
                          (config_tab_log['layer_type']==layer_type)&
                          (str(config_tab['config_type'])==str(config_type)),'q'] = q_value
    return config_tab, config_tab_log

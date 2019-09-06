import os
import pandas as pd
import numpy as np
import sys

from scipy import sparse

def get_count(tp, id): #id별로 그룹으로 묶고 그 그룹에 속한(같은 id를 갖는) 요소들의 갯수를 반환 ex) id=1 100개, id=2 200개...
    playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count

def filter_triplets(tp, min_uc=5, min_sc=0):
    # Only keep the triplets for items which were clicked on by at least min_sc users. 
    if min_sc > 0: #이 부분 어차피 실행 안됨(min_sc == 0 이기 때문에).
        itemcount = get_count(tp, 'movieId')
        tp = tp[tp['movieId'].isin(itemcount.index[itemcount >= min_sc])]  

    if min_uc > 0:
        usercount = get_count(tp, 'userId')
        tp = tp[tp['userId'].isin(usercount.index[usercount >= min_uc])]  #그룹화된 유저의 갯수가 5개 이상만 추출
    
    # Update both usercount and itemcount after filtering
    usercount, itemcount = get_count(tp, 'userId'), get_count(tp, 'movieId') 
    return tp, usercount, itemcount

def split_train_test_proportion(data, test_prop=0.2):  #각 사용자마다 20프로는 테스트데이터셋 80프로는 트레이닝 데이터셋을 분류
    data_grouped_by_user = data.groupby('userId')
    tr_list, te_list = list(), list()

    np.random.seed(98765)

    for i, (_, group) in enumerate(data_grouped_by_user): #i는 순서, _는 index of group, group은 userID별 group
        n_items_u = len(group) #그룹별 크기

        if n_items_u >= 5: #이미 그룹(userid)별 요소수는 다 5개 이상임 so else문은 실행 안댐
            idx = np.zeros(n_items_u, dtype='bool') 
            idx[np.random.choice(n_items_u, size=int(test_prop * n_items_u), replace=False).astype('int64')] = True

            tr_list.append(group[np.logical_not(idx)]) #np.logical_not(idx) false인것만 True로 (True는 false로 바꿈)
            te_list.append(group[idx])
        else:
            tr_list.append(group)

        if i % 1000 == 0:
            print("%d users sampled" % i)
            sys.stdout.flush()

    data_tr = pd.concat(tr_list) #list를 dataframe으로 변환(원래 concat은 dataframe 두개를 붙일때 사용)
    data_te = pd.concat(te_list)
    
    return data_tr, data_te

def numerize(tp, profile2id, show2id):
    
    uid = list(map(lambda x: profile2id[x], tp['userId'])) #profile2id는 unique한 userID와 순서가 저장되어 있음
    sid = list(map(lambda x: show2id[x], tp['movieId']))  #show2id 는 unique한 movieID와 순서가 저장되어 있음
  
    return pd.DataFrame(data={'uid': uid, 'sid': sid}, columns=['uid', 'sid'])

def data_preprocessing(data_dir):
    raw_data = pd.read_csv(os.path.join(data_dir, 'ratings.csv'), header=0)
    raw_data = raw_data[raw_data['rating'] > 3.5]
    print("[[raw_data]]\n",raw_data.head())

    raw_data, user_activity, item_popularity = filter_triplets(raw_data)

    sparsity = 1. * raw_data.shape[0] / (user_activity.shape[0] * item_popularity.shape[0]) 

    print("After filtering, there are %d watching events from %d users and %d movies (sparsity: %.3f%%)" % 
        (raw_data.shape[0], user_activity.shape[0], item_popularity.shape[0], sparsity * 100))

    unique_uid = user_activity.index #user's ID
    np.random.seed(98765)
    #순서 섞기
    idx_perm = np.random.permutation(unique_uid.size)
    unique_uid = unique_uid[idx_perm] 

    # create train/validation/test users
    n_users = unique_uid.size
    
    n_heldout_users = 10000

    tr_users = unique_uid[:(n_users - n_heldout_users * 2)]
    
    vd_users = unique_uid[(n_users - n_heldout_users * 2): (n_users - n_heldout_users)]
   
    te_users = unique_uid[(n_users - n_heldout_users):]

    train_plays = raw_data.loc[raw_data['userId'].isin(tr_users)]  #rawdata의 userID가 tr_users에 있으면 그 rawdata요소를 train_plays에 넣는다.

    unique_sid = pd.unique(train_plays['movieId']) #train_plays에서 unique한 movieID만 뽑아온다.

    show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid)) #enumberated: 반복문을 통해 튜플형태로 unique_sid의 인덱스와 순서를 반환한다.
    
    profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))

    pro_dir = os.path.join(data_dir, 'pro_sg')  #pro_sg 디렉터리 생성

    if not os.path.exists(pro_dir):
        os.makedirs(pro_dir)

    user_dir = os.path.join(data_dir, 'pro_sg/user_data')

    #user dir
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)

    with open(os.path.join(pro_dir, 'unique_sid.txt'), 'w') as f:
        for sid in unique_sid:
            f.write('%s\n' % sid)

    vad_plays = raw_data.loc[raw_data['userId'].isin(vd_users)] #vd_users가 raw_data의 userID가 있으면 그 요소들을 vad_plays에 넣어라

    vad_plays = vad_plays.loc[vad_plays['movieId'].isin(unique_sid)] #unique_sid가 vad_plays의 movieID가 있으면 그 요소들을 vad_plays에 넣어라

    vad_plays_tr, vad_plays_te = split_train_test_proportion(vad_plays)

    test_plays = raw_data.loc[raw_data['userId'].isin(te_users)]

    test_plays = test_plays.loc[test_plays['movieId'].isin(unique_sid)]

    test_plays_tr, test_plays_te = split_train_test_proportion(test_plays)
    
    #Save the data into (user_index, item_index) format
    train_data = numerize(train_plays, profile2id, show2id)


    train_data.to_csv(os.path.join(pro_dir, 'train.csv'), index=False)
   
    
    

    vad_data_tr = numerize(vad_plays_tr, profile2id, show2id)

    vad_data_tr.to_csv(os.path.join(pro_dir, 'validation_tr.csv'), index=False)

    vad_data_te = numerize(vad_plays_te, profile2id, show2id)

    vad_data_te.to_csv(os.path.join(pro_dir, 'validation_te.csv'), index=False)

    test_data_tr = numerize(test_plays_tr, profile2id, show2id)

    test_data_tr.to_csv(os.path.join(pro_dir, 'test_tr.csv'), index=False)

    test_data_te = numerize(test_plays_te, profile2id, show2id)

    test_data_te.to_csv(os.path.join(pro_dir, 'test_te.csv'), index=False)

def load_train_data(csv_file, n_items):
    tp = pd.read_csv(csv_file)
    n_users = tp['uid'].max() + 1
   
    rows, cols = tp['uid'], tp['sid']
    data = sparse.csr_matrix((np.ones_like(rows),
                             (rows, cols)), dtype='float64',
                             shape=(n_users, n_items))
    return data


def load_tr_te_data(csv_file_tr, csv_file_te, n_items):
    tp_tr = pd.read_csv(csv_file_tr)
    tp_te = pd.read_csv(csv_file_te)

    start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
    end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())

    rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['sid'] #어차피 validation user들은 train user들과 다른 user라서??
    rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['sid']

    data_tr = sparse.csr_matrix((np.ones_like(rows_tr),
                             (rows_tr, cols_tr)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
    data_te = sparse.csr_matrix((np.ones_like(rows_te),
                             (rows_te, cols_te)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
    return data_tr, data_te
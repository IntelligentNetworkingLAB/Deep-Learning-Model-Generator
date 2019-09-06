import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    #VAE arguments
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--data_dir', type=str, default='data/ml-20m', help='Movielens-20m dataset location')
    parser.add_argument('--model_dir', type=str, default='data/gModel', help='Stored Model location')
    parser.add_argument('--down_dir', type=str, default='download/dModel', help='Downloaded Model location')
    parser.add_argument('--preprocess', type=int, default=0, help='1 for preprocessing, 0 for no preprocessing')
    parser.add_argument('--batch_size', type=int, default=500, help='batch size')
    parser.add_argument('--batch_size_vad', type=int, default=2000, help='validation batch size')
    parser.add_argument('--total_anneal_steps', type=int, default=200000, help='number of total anneal steps')
    parser.add_argument('--anneal_cap', type=int, default=0.2, help='largest annealing parameter')
    parser.add_argument('--n_epochs', type=int, default=1000, help='training epochs')
    
    #FL arguments
    parser.add_argument('--n_participants', type=int, default=500, help="the number of participants for FL")
    parser.add_argument('--check_point', type=int, default=20, help="checking NDCG, Recall whenever 10 epochs")
    parser.add_argument('--user_IDs', type=list, default=[1,2,3,4,5,6,7,8,9,10], help="checking NDCG, Recall whenever 10 epochs")
    parser.add_argument('--n_RasPart', type=int, default=1, help="checking NDCG, Recall whenever 10 epochs")

    #[1,2,3,4,5,6,7,8,9,10]
    args = parser.parse_args()
    return args
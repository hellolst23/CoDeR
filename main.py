import os
import time
import torch
import argparse
import datetime
from plus.loss import Loss
from util.utils import *
import torch.optim as optim
from model.model import CoDeR
from torch.utils.data import DataLoader
from util.train_test import train, test
from data.dataset import load_data, RsData, sess_collate_fn
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2024, help='seed')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--plus', type=bool, default=True, help='enhenced model')
    parser.add_argument('--dataset', type=str, default='tmall', help='dataset name')
    parser.add_argument('--log_path', type=str, default='output.txt', help='log path')
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
    parser.add_argument('--epoch', type=int, default=50, help='the number of epochs to train for')
    parser.add_argument('--embedding_dim_i', type=int, default=128, help='item hidden state size')
    parser.add_argument('--embedding_dim_c', type=int, default=128, help='category hidden state size')
    parser.add_argument('--hidden_size', type=int, default=128, help='hidden state size ')
    parser.add_argument('--bias', action='store_false', help='bias in GNN linear layer')
    parser.add_argument('--n_demand', type=int, default=2, help='the dimension of demand space')
    parser.add_argument('--n_interest', type=int, default=2, help='the dimension of interest space')
    parser.add_argument('--n_gnn_layer', type=int, default=1, help='the number of gnn layer')
    parser.add_argument('--ad', action='store_true', help='whether adjustment')
    parser.add_argument('--lambda_catgy', type=int, default=2, help='category prediction weight')
    parser.add_argument('--predict_catgy_fun', type=str, default='dot', help='predict category method')
    parser.add_argument('--batch_norm', type=str, default='feature', help='which elements should be normalized')
    parser.add_argument('--rs', type=str, default='dot', help='pvsd calculation method')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--embed_l2', type=float, default=1e-5, help='l2 penalty')
    parser.add_argument('--drop_item', type=float, default=0, help='item dropout')
    parser.add_argument('--drop_catgy', type=float, default=0.5, help='category dropout')
    parser.add_argument('--alpha', type=float, default=0.85, help='critical value for modularity gain')
    parser.add_argument('--patience', type=int, default=5, help='the number of epoch to wait before early stop')
    parser.add_argument('--validation', action='store_true', help='validation')
    parser.add_argument('--valid_portion', type=float, default=0.2, help='split the portion of training set as validation set')
    opt = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    time_path = time.strftime("%y%m%d-%H%M%S", time.localtime(time.time()))
    log_dir_train = os.path.join('./log', opt.dataset, f"{time_path}")
    log_path = os.path.join(log_dir_train, "output.txt")
    opt.log_path = log_path
    if not os.path.exists(log_dir_train):
        os.makedirs(log_dir_train)

    seed_torch(log_path, opt.seed)
    file_write(log_path,
               f"----------------------------------- Start Loading data... @ {datetime.datetime.now()} -----------------------------------")
    dataset_dir = os.path.join('datasets', opt.dataset)
    train_data, valid_data, test_data, candidate_category, session_info = load_data(dataset_dir,
                                                                               validation_flag=opt.validation,
                                                                               valid_portion=opt.valid_portion)
    file_write(log_path,'-' * 45 + "Finish Loading data... @ %ss " % datetime.datetime.now() + '-' * 45)
    file_write(log_path, f"opt.dataset:  {opt.dataset}")

    train_data = RsData(log_path, train_data)
    test_data = RsData(log_path, test_data)
    valid_data = RsData(log_path, valid_data)
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, num_workers=16, shuffle=True,
                              collate_fn=sess_collate_fn)
    test_loader = DataLoader(test_data, batch_size=opt.batch_size, num_workers=16, shuffle=True,
                             collate_fn=sess_collate_fn)
    valid_loader = DataLoader(valid_data, batch_size=opt.batch_size, num_workers=16, shuffle=True,
                             collate_fn=sess_collate_fn)
    
    try:
        opt.n_node = session_info["num_of_item[not include 0]"] + 1
    except:
        opt.n_node = session_info["num_of_item"] + 1
    opt.n_node_c = session_info["num_of_category"] + 1

    model = CoDeR(opt.n_node, opt.n_node_c, opt)
    model.to(device)
    criterion = Loss(opt)
    rec_optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    Ks = [10,20,40,50,60,80,100]
    Ks_auc = [50,100,200,500]

    best_result_recall = [0] * len(Ks)
    best_result_mrr = [0] * len(Ks)
    best_result_ndcg = [0] * len(Ks)
    best_result_auc = [0] * len(Ks_auc)

    best_epoch_auc = [0] * len(Ks_auc)
    best_epoch_ndcg = [0] * len(Ks)
    best_epoch_recall = [0] * len(Ks)
    best_epoch_mrr = [0] * len(Ks)
    start = time.time()

    bad_counter = 0
    for epoch in range(opt.epoch):
        file_write(log_path, f'epoch: {epoch}')
        train(log_path, model, criterion, rec_optimizer, train_loader, candidate_category, device, epoch, opt)
        recall, mrr, ndcg, auc = test(opt, model, criterion, valid_loader, candidate_category, device, Ks)
        flag = 0
        for i, topk in enumerate(Ks):
            if recall[i] >= best_result_recall[i]:
                best_result_recall[i] = recall[i]
                best_epoch_recall[i] = epoch
                flag = 1
            if mrr[i] >= best_result_mrr[i]:
                best_result_mrr[i] = mrr[i]
                best_epoch_mrr[i] = epoch
                flag = 1
            if ndcg[i] >= best_result_ndcg[i]:
                best_result_ndcg[i] = ndcg[i]
                best_epoch_ndcg[i] = epoch
                if topk == 40:
                    print("saving...")
                    ckpt_dict = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': rec_optimizer.state_dict()
                    }
                    torch.save(ckpt_dict, os.path.join(log_dir_train,f"model.pth.tar"))
                flag = 1
        for i, k in enumerate(Ks_auc):
            if auc[i] >= best_result_auc[i]:
                best_result_auc[i] = auc[i]
                best_epoch_auc[i] = epoch
                flag = 1

        cprint(log_path, f'Current Result Epoch {epoch}:')
        print_result(log_path, 'Recall', recall,Ks)
        print_result(log_path, 'MRR', mrr,Ks)
        print_result(log_path, 'NDCG', ndcg, Ks)
        print_auc_result(log_path, 'AUC', auc, Ks_auc)

        bad_counter += 1 - flag

        if bad_counter >= opt.patience:
            print("loading...")
            ckpt = torch.load(os.path.join(log_dir_train,f"model.pth.tar"))
            model.load_state_dict(ckpt['state_dict'])
            t_recall, t_mrr, t_ndcg, t_auc = test(opt, model, criterion, test_loader, candidate_category, device, Ks)
            cprint(log_path, f'test result')
            print_result(log_path, 'Recall', t_recall, Ks)
            print_result(log_path, 'MRR', t_mrr, Ks)
            print_result(log_path, 'NDCG', t_ndcg, Ks)
            print_auc_result(log_path, 'AUC', t_auc, Ks_auc)
            break
        torch.cuda.empty_cache()
    file_write(log_path, '-------------------------------------------------------')
    end = time.time()
    file_write(log_path, f"Run time: {(end - start) / 60.0}min")


if __name__ == '__main__':
    main()
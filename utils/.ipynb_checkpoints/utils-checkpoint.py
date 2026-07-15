import numpy as np
import torch
import os
import random
import csv
import pickle
import torch.nn as nn
from torch.utils.data import Dataset
import subprocess


class SeqEncoder:
    def __init__(self):
        self.categories = np.array(list('ACDEFGHIKLMNPQRSTVWYX'))

    def char_to_one_hot_encoding(self, c):
        X_int = np.zeros(len(self.categories), dtype=np.int8)
        index = np.where(self.categories == c)[0]

        if len(index) == 0: return X_int

        X_int[index] = 1
        return X_int

    def seq_to_one_hot_encoding(self, seq):
        return np.array([self.char_to_one_hot_encoding(c) for c in seq])

    def one_hot_encoding_to_seq(self, one_hot_encoding):
        X = np.array(one_hot_encoding).reshape(-1, 21)
        last_index = np.where(np.sum(X, axis=1) == 0)[0]
        last_index = 1000 if len(last_index) == 0 else last_index[0]
        return ''.join(self.categories[X.argmax(axis=1)][:last_index])
    
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)
    
class NumpyPairDataset(Dataset):
    def __init__(self, np_array):
        self.data = np_array

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)
        return x, y
    
def format_esm(a):
    if type(a) == dict:
        a = a['mean_representations'][33]
    return a
    
def load_esm(lookup):
    esm = format_esm(torch.load('./esm_data/' + lookup + '.pt'))
    return esm.unsqueeze(0)

def get_cluster_center(model_emb, ec_id_dict):
    cluster_center_model = {}
    id_counter = 0
    with torch.no_grad():
        for ec in (list(ec_id_dict.keys())):
            ids_for_query = list(ec_id_dict[ec])
            id_counter_prime = id_counter + len(ids_for_query)
            emb_cluster = model_emb[id_counter: id_counter_prime]
            cluster_center = emb_cluster.mean(dim=0)
            cluster_center_model[ec] = cluster_center.detach().cpu()
            id_counter = id_counter_prime
    return cluster_center_model
    
def model_embedding_test(id_ec_test, model, device, dtype):
    ids_for_query = list(id_ec_test.keys())
    esm_to_cat = [load_esm(id) for id in ids_for_query]
    esm_emb = torch.cat(esm_to_cat).to(device=device, dtype=dtype)
    model_emb = model(esm_emb)
    return model_emb

def retrive_esm1b_embedding(fasta_name):
    esm_script = "./extract.py"
    esm_out = "./esm_data"
    esm_type = "esm1b_t33_650M_UR50S"
    fasta_name = "data/" + fasta_name + ".fasta"
    python_path = "/home/jeons9/miniconda3/envs/ECPICK/bin/python"
    command = [python_path, esm_script, esm_type, 
              fasta_name, esm_out, "--include", "mean"]
    subprocess.run(command)

def retrive_esm1b_embedding_inference(fasta_name):
    esm_script = "./extract.py"
    esm_out = "./esm_data"
    esm_type = "esm1b_t33_650M_UR50S"
    fasta_name = "clean_inference/" + fasta_name + ".fasta"
    python_path = "/home/jeons9/miniconda3/envs/ECPICK/bin/python"
    command = [python_path, esm_script, esm_type, 
              fasta_name, esm_out, "--include", "mean"]
    subprocess.run(command)

def prepare_infer_fasta(fasta_name):
    # retrive_esm1b_embedding(fasta_name)
    retrive_esm1b_embedding_inference(fasta_name)
    csvfile = open('clean_inference/' + fasta_name +'.csv', 'w', newline='')
    csvwriter = csv.writer(csvfile, delimiter = '\t')
    csvwriter.writerow(['Entry', 'EC number', 'Sequence'])
    fastafile = open('clean_inference/' + fasta_name +'.fasta', 'r')
    for i in fastafile.readlines():
        if i[0] == '>':
            csvwriter.writerow([i.strip()[1:], ' ', ' '])

def dist_map_helper(keys1, lookup1, keys2, lookup2):
    dist = {}
    for i, key1 in (enumerate(keys1)):
        current = lookup1[i].unsqueeze(0)
        dist_norm = (current - lookup2).norm(dim=1, p=2)
        dist_norm = dist_norm.detach().cpu().numpy()
        dist[key1] = {}
        for j, key2 in enumerate(keys2):
            dist[key1][key2] = dist_norm[j]
    return dist

def get_pred_labels(out_filename, pred_type="_maxsep", date=None):
    with open(f'./utils/output_classes.pkl', 'rb') as f :
        file = pickle.load(f)
    out_filename = out_filename.replace('inputsinputs', 'inputs')
    file_name = out_filename
    result = open(file_name+'.csv', 'r')
    csvreader = csv.reader(result, delimiter=',')
    pred_label = []
    for row in csvreader:
        preds_ec_lst = []
        preds_with_dist = row[1:]
        for pred_ec_dist in preds_with_dist:
            # get EC number 3.5.2.6 from EC:3.5.2.6/10.8359
            ec_i = pred_ec_dist.split(":")[1].split("/")[0]
            preds_ec_lst.append(ec_i)
        pred_label.append(preds_ec_lst)

    return pred_label

def get_pred_probs(out_filename, pred_type="_maxsep"):
    out_filename = out_filename.replace('inputsinputs', 'inputs')
    file_name = out_filename
    result = open(file_name+'.csv', 'r')
    csvreader = csv.reader(result, delimiter=',')
    pred_probs = []
    for row in csvreader:
        preds_ec_lst = []
        preds_with_dist = row[1:]
        probs = torch.zeros(len(preds_with_dist))
        count = 0
        for pred_ec_dist in preds_with_dist:
            # get EC number 3.5.2.6 from EC:3.5.2.6/10.8359
            ec_i = float(pred_ec_dist.split(":")[1].split("/")[1])
            probs[count] = ec_i
            #preds_ec_lst.append(probs)
            count += 1
        # sigmoid of the negative distances 
        probs = (1 - torch.exp(-1/probs)) / (1 + torch.exp(-1/probs))
        probs = probs/torch.sum(probs)
        pred_probs.append(probs)
    return pred_probs

def get_dist_map_test(model_emb_train, model_emb_test,
                      ec_id_dict_train, id_ec_test,
                      device, dtype, dot=False):
    # print("The embedding sizes for train and test:",
          # model_emb_train.size(), model_emb_test.size())
    # get cluster center for all EC appeared in training set
    cluster_center_model = get_cluster_center(
        model_emb_train, ec_id_dict_train)
    total_ec_n, out_dim = len(ec_id_dict_train.keys()), model_emb_train.size(1)
    model_lookup = torch.zeros(total_ec_n, out_dim, device=device, dtype=dtype)
    ecs = list(cluster_center_model.keys())
    for i, ec in enumerate(ecs):
        model_lookup[i] = cluster_center_model[ec]
    model_lookup = model_lookup.to(device=device, dtype=dtype)
    # calculate distance map between n_query_test * total_ec_n (training) pairs
    ids = list(id_ec_test.keys())
    # print(f'Calculating eval distance map, between {len(ids)} test ids '
          # f'and {total_ec_n} train EC cluster centers')
    if dot:
        eval_dist = dist_map_helper_dot(ids, model_emb_test, ecs, model_lookup)
    else:
        eval_dist = dist_map_helper(ids, model_emb_test, ecs, model_lookup)
    return eval_dist

def seed_everything(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
def get_ec_id_dict(csv_name: str) -> dict:
    csv_file = open(csv_name)
    csvreader = csv.reader(csv_file, delimiter='\t')
    id_ec = {}
    ec_id = {}

    for i, rows in enumerate(csvreader):
        if i > 0:
            id_ec[rows[0]] = rows[1].split(';')
            for ec in rows[1].split(';'):
                if ec not in ec_id.keys():
                    ec_id[ec] = set()
                    ec_id[ec].add(rows[0])
                else:
                    ec_id[ec].add(rows[0])
    return id_ec, ec_id

def maximum_separation(dist_lst, first_grad, use_max_grad):
    opt = 0 if first_grad else -1
    gamma = np.append(dist_lst[1:], np.repeat(dist_lst[-1], 10))
    sep_lst = np.abs(dist_lst - np.mean(gamma))
    sep_grad = np.abs(sep_lst[:-1]-sep_lst[1:])
    if use_max_grad:
        # max separation index determined by largest grad
        max_sep_i = np.argmax(sep_grad)
    else:
        # max separation index determined by first or the last grad
        large_grads = np.where(sep_grad > np.mean(sep_grad))
        max_sep_i = large_grads[-1][opt]
    # if no large grad is found, just call first EC
    if max_sep_i >= 5:
        max_sep_i = 0
    return max_sep_i

def infer_confidence_gmm(distance, gmm_lst):
    confidence = []
    for j in range(len(gmm_lst)):
        main_GMM = gmm_lst[j]
        a, b = main_GMM.means_
        true_model_index = 0 if a[0] < b[0] else 1
        certainty = main_GMM.predict_proba([[distance]])[0][true_model_index]
        confidence.append(certainty)
    return np.mean(confidence)

def write_max_sep_choices(df, csv_name, first_grad=True, use_max_grad=False, gmm = None):
    out_file = open(f'{csv_name}'+ '.csv', 'w', newline='')
    csvwriter = csv.writer(out_file, delimiter=',')
    
    all_test_EC = set()
    for col in df.columns:
        ec = []
        smallest_10_dist_df = df[col].nsmallest(10)
        dist_lst = list(smallest_10_dist_df)
        max_sep_i = maximum_separation(dist_lst, first_grad, use_max_grad)
        for i in range(max_sep_i+1):
            EC_i = smallest_10_dist_df.index[i]
            dist_i = smallest_10_dist_df.iloc[i]
            if gmm != None:
                gmm_lst = pickle.load(open(gmm, 'rb'))
                dist_i = infer_confidence_gmm(dist_i, gmm_lst)
            dist_str = "{:.4f}".format(dist_i)
            all_test_EC.add(EC_i)
            ec.append('EC:' + str(EC_i) + '/' + dist_str)
        ec.insert(0, col)
        csvwriter.writerow(ec)
    return


def create_soft_label_tensor(pred_labels, pred_probs, ec_map, num_classes, device, dtype):
    row_indices = []  
    col_indices = []  
    values = []       

    batch_size = len(pred_labels)

    for batch_idx, (label_names, prob_tensor) in enumerate(zip(pred_labels, pred_probs)):
        current_probs = prob_tensor.tolist()
        for ec_name, p_val in zip(label_names, current_probs):
            if ec_name in ec_map:
                row_indices.append(batch_idx)             
                col_indices.append(ec_map[ec_name]) 
                values.append(p_val)                      

    rows_t = torch.tensor(row_indices, dtype=torch.long, device=device)
    cols_t = torch.tensor(col_indices, dtype=torch.long, device=device)
    vals_t = torch.tensor(values, dtype=dtype, device=device)

    clean_outputs = torch.zeros((batch_size, num_classes), dtype=dtype, device=device)
    clean_outputs.index_put_((rows_t, cols_t), vals_t)
    
    return clean_outputs
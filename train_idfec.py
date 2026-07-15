import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import os
import glob
import argparse
import cv2
import pickle
import pandas as pd
import numpy as np
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset, Subset, ConcatDataset
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import f1_score
from torch.utils.tensorboard import SummaryWriter
from utils.utils import *
from models.IDFEC import IDF_EC
from models.ECPICK import PredictionModel
from models.HITEC import Model
from models.CLEAN import LayerNormNet


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class Trainer:
    def __init__(self, gate, model_choice):
        self.device = None
        self.gate = gate
        self.classes = None
        self.train_loss_list = []
        self.val_loss_list =[]
        self.train_f1_micro_list = []
        self.train_f1_macro_list = []
        self.val_f1_micro_list = []
        self.val_f1_macro_list = []
        
        self.ecpick_model_path = glob.glob(os.path.join(f"./base_learners/ECPICK_models", '*.pth'))
        self.hitec_model_path = f"./base_learners/HIT-EC_models/hitec.ckpt"
        self.clean_model_path = f"./base_learners/CLEAN_models/clean.pth"
        self.clean_emb_path = f"./base_learners/CLEAN_models/STACKING_train_esm_embedding_cache.pt"
        self.output_classes_path = f"./utils/output_classes.pkl"
        self.model_save_path = f'./saved_models/'

        self.model_choice = model_choice
        
        
        print("-##### TRAIN STARTED #####-")
        with open(self.output_classes_path, 'rb') as f:
            self.classes = pickle.load(f)    
        print(f"-##### PREDICTABLE LABEL NUMBER : {len(self.classes[3])} #####-")
        
        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available:
            print(f"-##### CUDA is available #####-")
            self.device = torch.cuda.current_device()
        else:
            print(f"-##### CUDA is not available #####-")
            self.device = torch.device('cpu')
        
        
        ##### ECPICK  #####
        self.ecpick_models = []
        for path in self.ecpick_model_path:
            ecpick_model = PredictionModel(
                output_classes=self.classes,
                cuda_support=self.cuda_available,
                cuda_device=self.device,
                dropout_rate=0.8,
                relu_size=384,
                beta=0.6,
                model_path=path
            )
            ecpick_model = ecpick_model.to(self.device)
            ecpick_model.eval()
            self.ecpick_models.append(ecpick_model)
        
        ##### HIT-EC  #####
        config = {
            'ah': 2,
            'dr': 0.1,
            'beta': 0.59,
            'dimension': 1024,
            'output_dims': [len(self.classes[0]), len(self.classes[1]), len(self.classes[2]), len(self.classes[3])]
        }
        self.hitec_model = Model(
            config=config
        )
        model_data = torch.load(self.hitec_model_path, map_location=torch.device('cuda', self.device))
        # self.hitec_model.load_state_dict(model_data['callbacks']['StochasticWeightAveraging']['average_model_state'])state_dict
        self.hitec_model.load_state_dict(model_data['state_dict'])
        self.hitec_model.to(self.device)
        self.hitec_model.eval()
        
        ##### CLEAN #####
        self.dtype = torch.float32
        # self.ec_id_dict_train = Non
        checkpoint = torch.load(self.clean_model_path, map_location=torch.device('cuda', self.device))
        self.clean_model = LayerNormNet(512, 256, self.device, self.dtype)
        self.clean_model.load_state_dict(checkpoint)
        self.clean_model.eval()
        self.emb_train = torch.load(self.clean_emb_path, map_location=torch.device('cuda', self.device))
        self.ec_to_idx_map = {ec: i for i, ec in enumerate(self.classes[3])}
        
        ##### STACKING #####
        self.stacking_model = IDF_EC(
            output_classes=self.classes,
            cuda_available=self.cuda_available,
            device=self.device,
            num_classes=len(self.classes[3]),
            ecpick_feature=384,
            hitec_feature=1024,
            clean_feature=256
        ).to(self.device)

        if self.gate == True:
            checkpoint_path = f'./saved_models/{self.model_choice}.pth'
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda', self.device))
            self.stacking_model.load_state_dict(checkpoint, strict=False)
        
        os.makedirs(self.model_save_path, exist_ok=True)
        
    def aggregate_and_print_stats(self, stats_list, title="Stats"):
        if not stats_list:
            return

        agg_data = {}
        for k in stats_list[0].keys():
            agg_data[k] = [d.get(k, 0.0) for d in stats_list]

        tensor_groups = sorted(set(
            k.replace("_mean","").replace("_std","").replace("_min","").replace("_max","")
            for k in stats_list[0].keys()
            if k.endswith(("_mean", "_std", "_min", "_max"))
        ))

        scalar_keys = sorted([
            k for k in stats_list[0].keys()
            if not k.endswith(("_mean", "_std", "_min", "_max"))
        ])

        print("\n" + f"=== {title} (Epoch Summary) ===")
        print(f"{'LOGIT NAME':<24} | {'MEAN':<8} | {'STD(Avg)':<8} | {'MIN(Global)':<11} | {'MAX(Global)':<11}")
        print("-" * 80)

        for name in tensor_groups:
            mean_v = np.mean(agg_data.get(f"{name}_mean", [0.0]))
            std_v  = np.mean(agg_data.get(f"{name}_std",  [0.0]))
            min_v  = np.min( agg_data.get(f"{name}_min",  [0.0]))
            max_v  = np.max( agg_data.get(f"{name}_max",  [0.0]))

            print(f"{name:<24} | {mean_v:>8.4f} | {std_v:>8.4f} | {min_v:>11.4f} | {max_v:>11.4f}")

        if scalar_keys:
            print("-" * 80)
            for k in scalar_keys:
                vals = np.array(agg_data[k], dtype=np.float32)
                mean_v = float(np.mean(vals))
                std_v  = float(np.std(vals))
                min_v  = float(np.min(vals))
                max_v  = float(np.max(vals))
                print(f"{k:<24} | {mean_v:>8.4f} | {std_v:>8.4f} | {min_v:>11.4f} | {max_v:>11.4f}")

        print("=" * 80 + "\n")
        
        
    
    def train(self, ecpick_train_loader, hitec_train_loader, clean_train_loader, clean_val_loader, ecpick_val_loader, hitec_val_loader, ecpick_models, hitec_model, clean_model, stacking_model, epochs, clean_batches_indices_train, clean_batches_indices_val, optimizer, gate):
        total_train_loss = []
        total_val_loss = []
        total_train_f1_micro = []
        total_train_f1_macro = []
        total_val_f1_micro = []
        total_val_f1_macro = []
        
        loss_fn = nn.BCEWithLogitsLoss()
        
        for epoch in range(45):
            
            train_stats_accumulator = []
            train_loss = 0.0
            combined_loader = zip(ecpick_train_loader, hitec_train_loader, clean_batches_indices_train)
            
            train_preds = []
            train_labels = []
            stacking_model.train()
            for batch, ((ecpick_seqs, ecpick_labels), (hitec_seqs, hitec_labels), clean_indices) in enumerate(combined_loader):
                ecpick_features = []
                ecpick_outputs =[]
                ecpick_labels = ecpick_labels.to(self.device).float()

                ecpick_seqs = ecpick_seqs.to(self.device)
                hitec_seqs = hitec_seqs.to(self.device)
                with torch.no_grad():
                    ##### ECPICK #####
                    for ecpick_model in ecpick_models:
                        ecpick_result = ecpick_model(ecpick_seqs)
                        ecpick_features.append(ecpick_result['feature_vector'])
                        ecpick_outputs.append(ecpick_result['final_output'])

                    ecpick_stacked_outputs = torch.stack(ecpick_outputs, dim=0)   
                    ecpick_mean_outputs = ecpick_stacked_outputs.mean(dim=0) 
                    ecpick_stacked_features = torch.stack(ecpick_features, dim=0) 
                    ecpick_mean_feature = ecpick_stacked_features.mean(dim=0)

                    ##### HIT-EC #####
                    hitec_result = hitec_model(hitec_seqs.long())
                    hitec_features = hitec_result['feature_vectors']
                    hitec_outputs = hitec_result['final']
                    hitec_outputs = hitec_outputs[:, -(len(self.classes[3])):]

                    hitec_features = [feat.to(self.device) for feat in hitec_features]
                    
                    ##### CLEAN #####
                    batch_df = clean_train_loader.iloc[clean_indices]
                    batch_df.to_csv(f"./clean_inference/train.csv", index=False)
                   
                    id_ec_test, ec_id_dict_test = get_ec_id_dict(f'./clean_inference/train.csv')
                    emb_test = model_embedding_test(id_ec_test, clean_model, self.device, self.dtype)
                    clean_features = emb_test
                    eval_dist = get_dist_map_test(self.emb_train, emb_test, self.ec_id_dict_train, id_ec_test, self.device, self.dtype)
                    
                    
                    eval_df = pd.DataFrame.from_dict(eval_dist)
                    write_max_sep_choices(eval_df, f"./clean_inference/train", gmm=None)
                    
                    pred_label = get_pred_labels(f"./clean_inference/train", pred_type='_maxsep')
                    pred_probs = get_pred_probs(f"./clean_inference/train", pred_type='_maxsep')


                    clean_outputs = create_soft_label_tensor(pred_label, pred_probs, self.ec_to_idx_map, len(self.classes[3]), self.device, self.dtype)

                stacking_output, batch_stats  = stacking_model(
                    ecpick_mean_feature, 
                    ecpick_mean_outputs, 
                    hitec_features, 
                    hitec_outputs,
                    clean_features,
                    clean_outputs,
                    gate=gate
                )
                
                
                train_stats_accumulator.append(batch_stats)
                
                loss = loss_fn(stacking_output, ecpick_labels[:, -len(self.classes[3]):]) 
                outputs = (torch.sigmoid(stacking_output) > 0.22).int()
                train_preds.append(outputs.cpu().numpy())
                train_labels.append(ecpick_labels[:, -len(self.classes[3]):].cpu().numpy())


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                

            self.aggregate_and_print_stats(train_stats_accumulator, title=f"Epoch {epoch+1} TRAIN")
                
            train_loss = train_loss / len(ecpick_train_loader)

            
            total_train_loss.append(train_loss)
            
            train_labels = np.vstack(train_labels)
            train_preds = np.vstack(train_preds)
            train_f1_macro = f1_score(train_labels, train_preds, average='macro', zero_division=0)
            train_f1_micro = f1_score(train_labels, train_preds, average='micro', zero_division=0)
            
            
            combined_val_loader = zip(ecpick_val_loader, hitec_val_loader, clean_batches_indices_val)
            
            val_stats_accumulator = []
            val_preds=[]
            val_labels=[]
            val_loss = 0.0
            stacking_model.eval()
            with torch.no_grad():
                for batch, ((ecpick_seqs, ecpick_labels), (hitec_seqs, hitec_labels), clean_indices) in enumerate(combined_val_loader):
                    ecpick_features = []
                    ecpick_outputs =[]
                    ecpick_labels = ecpick_labels.to(self.device).float()

                    ecpick_seqs = ecpick_seqs.to(self.device)
                    hitec_seqs = hitec_seqs.to(self.device)
                
                    ##### ECPICK #####
                    for ecpick_model in ecpick_models:
                        ecpick_result = ecpick_model(ecpick_seqs)
                        ecpick_features.append(ecpick_result['feature_vector'])
                        ecpick_outputs.append(ecpick_result['final_output'])

                    ecpick_stacked_outputs = torch.stack(ecpick_outputs, dim=0)   
                    ecpick_mean_outputs = ecpick_stacked_outputs.mean(dim=0) 
                    ecpick_stacked_features = torch.stack(ecpick_features, dim=0) 
                    ecpick_mean_feature = ecpick_stacked_features.mean(dim=0)

                    ##### HIT-EC #####
                    hitec_result = hitec_model(hitec_seqs.long())
                    hitec_features = hitec_result['feature_vectors']
                    hitec_outputs = hitec_result['final']
                    hitec_outputs = hitec_outputs[:, -(len(self.classes[3])):]
                    hitec_features = [feat.to(self.device) for feat in hitec_features]
                    
                    ##### CLEAN #####
                    batch_df = clean_val_loader.iloc[clean_indices]
                    batch_df.to_csv(f"./clean_inference/val.csv", index=False)
                    
                    id_ec_test, ec_id_dict_test = get_ec_id_dict(f'./clean_inference/val.csv')
                    emb_test = model_embedding_test(id_ec_test, clean_model, self.device, self.dtype)
                    clean_features = emb_test
                    eval_dist = get_dist_map_test(self.emb_train, emb_test, self.ec_id_dict_train, id_ec_test, self.device, self.dtype)
                    
                    
                    eval_df = pd.DataFrame.from_dict(eval_dist)
                    write_max_sep_choices(eval_df, f"./clean_inference/val", gmm=None)
                    
                    pred_label = get_pred_labels(f"./clean_inference/val", pred_type='_maxsep')
                    pred_probs = get_pred_probs(f"./clean_inference/val", pred_type='_maxsep')
                    

                    clean_outputs = create_soft_label_tensor(pred_label, pred_probs, self.ec_to_idx_map, len(self.classes[3]), self.device, self.dtype)
                  
                    
                    
                    stacking_output, batch_stats  = stacking_model(
                        ecpick_mean_feature, 
                        ecpick_mean_outputs, 
                        hitec_features, 
                        hitec_outputs,
                        clean_features,
                        clean_outputs,
                        gate
                    )  

                    val_stats_accumulator.append(batch_stats)
                    loss = loss_fn(stacking_output, ecpick_labels[:, -len(self.classes[3]):])
                    
                    outputs = (torch.sigmoid(stacking_output) > 0.22).int()
                    val_preds.append(outputs.cpu().numpy())
                    val_labels.append(ecpick_labels[:, -len(self.classes[3]):].cpu().numpy())

                    val_loss += loss.item()
                    
            val_loss = val_loss / len(ecpick_val_loader)
            

            self.aggregate_and_print_stats(val_stats_accumulator, title=f"Epoch {epoch+1} VALID")
            
            total_val_loss.append(val_loss)
            
            val_labels = np.vstack(val_labels)
            val_preds = np.vstack(val_preds)
            val_f1_macro = f1_score(val_labels, val_preds, average='macro', zero_division=0)
            val_f1_micro = f1_score(val_labels, val_preds, average='micro', zero_division=0)
            
            if gate == False:
                torch.save(stacking_model.state_dict(), self.model_save_path + 'stacking_epoch=' + str(epoch + 1) +'.pth')
            else:
                torch.save(stacking_model.state_dict(), self.model_save_path + 'final_epoch=' + str(epoch + 1) +'.pth')
            
            print(f"-#####   Epochs [{epoch + 1} | {epochs}]\tTRAIN LOSS : {train_loss:.4f}\tTRAIN F1 MICRO : {train_f1_micro:.4f}\tTRAIN F1 MACRO : {train_f1_macro:.4f}   #####-")
            print(f"-#####   Epochs [{epoch + 1} | {epochs}]\tVALIDATION LOSS : {val_loss:.4f}\tVALIDATION F1 MICRO : {val_f1_micro:.4f}\tVALIDATION F1 MACRO : {val_f1_macro:.4f}   #####-")
                    
    def get_batches_indices(self, indices, batch_size):
        return [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]
    
    def run(self):
        if self.gate == False:
            initial_epochs = 40
            learning_rate = 0.000045
        else:
            initial_epochs = 3
            learning_rate = 0.0000015
        batch_size = 128
        
        g = torch.Generator()
        g.manual_seed(42)
        
        # Dataset Loading
        train_dataset_path = f"./data/ECPICK/ecpick_train.npy"
        val_dataset_path = f"./data/ECPICK/ecpick_val.npy"
        print(f"\n-##### ECPICK DATASET LOADING #####-")
        train_dataset = (np.load(train_dataset_path, allow_pickle=True))
        train_dataset = NumpyPairDataset(train_dataset)
        val_dataset = (np.load(val_dataset_path, allow_pickle=True))
        val_dataset = NumpyPairDataset(val_dataset)
        
        train_indices = torch.randperm(len(train_dataset), generator=g).tolist()
        val_indices = torch.randperm(len(val_dataset), generator=g).tolist()
        
        ecpick_train_sampler = torch.utils.data.Subset(train_dataset, train_indices)
        ecpick_val_sampler = torch.utils.data.Subset(val_dataset, val_indices)
        
        ecpick_train_loader = DataLoader(ecpick_train_sampler, batch_size=batch_size, shuffle=False, num_workers=1)
        ecpick_val_loader   = DataLoader(ecpick_val_sampler, batch_size=batch_size, shuffle=False, num_workers=1)
        print(f'-##### ECPICK Train Dataset Size: {len(train_dataset)} #####-')
        print(f'-##### ECPICK Val Dataset Size: {len(val_dataset)} #####-')
        
        print(f"\n-##### HIT-EC DATASET LOADING #####-")
        with open(f'./utils/tokenizer.pickle', 'rb') as f:
            tokenizer = pickle.load(f)
            
        train_dataset_path = f"./data/HIT-EC/hitec_train.npy"
        val_dataset_path = f"./data/HIT-EC/hitec_val.npy"
        train_dataset = (np.load(train_dataset_path, allow_pickle=True))
        train_dataset = NumpyPairDataset(train_dataset)
        val_dataset = (np.load(val_dataset_path, allow_pickle=True))
        val_dataset = NumpyPairDataset(val_dataset)
        
        hitec_train_sampler = torch.utils.data.Subset(train_dataset, train_indices)
        hitec_val_sampler = torch.utils.data.Subset(val_dataset, val_indices)

        hitec_train_loader = DataLoader(hitec_train_sampler, batch_size=batch_size, shuffle=False, num_workers=1)
        hitec_val_loader   = DataLoader(hitec_val_sampler, batch_size=batch_size, shuffle=False, num_workers=1)
        print(f'-##### HIT-EC Train Dataset Size: {len(train_dataset)} #####-')
        print(f'-##### HIT-EC Val Dataset Size: {len(val_dataset)} #####-')
        
        clean_batches_indices_train = self.get_batches_indices(train_indices, batch_size)
        clean_batches_indices_val   = self.get_batches_indices(val_indices, batch_size)
       
        clean_train_dataset_path = f"./data/CLEAN/train_clean_final.csv" 
        self.id_ec_train, self.ec_id_dict_train = get_ec_id_dict(f"./data/CLEAN/train_clean_final.csv" )
        clean_train_df = pd.read_csv(clean_train_dataset_path)
        clean_val_dataset_path = f"./data/CLEAN/val_clean_final.csv"
        clean_val_df = pd.read_csv(clean_val_dataset_path)
        print(f"\n-##### CLEAN Train Dataset Size : {len(clean_train_df)} #####-")
        print(f"-##### CLEAN Val Dataset Size : {len(clean_val_df)} #####-\n")
        
        optimizer = torch.optim.AdamW(self.stacking_model.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        
        
        self.train(
            ecpick_train_loader=ecpick_train_loader,
            hitec_train_loader=hitec_train_loader,
            clean_train_loader=clean_train_df,
            clean_val_loader=clean_val_df,
            ecpick_val_loader=ecpick_val_loader,
            hitec_val_loader=hitec_val_loader,
            ecpick_models=self.ecpick_models, 
            hitec_model=self.hitec_model,
            clean_model=self.clean_model,
            stacking_model=self.stacking_model,
            epochs=initial_epochs,
            clean_batches_indices_train=clean_batches_indices_train,
            clean_batches_indices_val=clean_batches_indices_val,
            optimizer=optimizer,
            gate=self.gate
        )
        

        
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gate', 
                        type=lambda x: str(x).lower() in ['true', '1'], 
                        choices=[True, False], 
                        default=False, 
                        help="Enable or disable the gate (True/False)")
    parser.add_argument('--model_config', 
                        type=str, 
                        default=None, 
                        help="Select the best performance model to train the dynamic fusion gate")
    args = parser.parse_args()

    if args.gate and args.model_config is None:
        parser.error("--model is required when --gate is True!")
    
    gate = args.gate
    model_choice = args.model_config
    trainer = Trainer(gate=gate, model_choice=model_choice)
    trainer.run()
    
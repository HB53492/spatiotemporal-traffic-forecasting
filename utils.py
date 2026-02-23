import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import kagglehub
from sklearn.preprocessing import StandardScaler

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def get_parameter_groups(model, weight_decay=1e-4):
    decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)

    groups = []
    if len(decay) > 0:
        groups.append({'params': decay, 'weight_decay': weight_decay})
    if len(no_decay) > 0:
        groups.append({'params': no_decay, 'weight_decay': 0.0})

    return groups

def normalize_adj(adj, eps=1e-5):
    device = adj.device
    A = adj + torch.eye(adj.size(0), device=device)
    
    D = A.sum(-1)
    D_inv_sqrt = torch.pow(D + eps, -0.5)
    D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0
    
    return D_inv_sqrt[:, None] * A * D_inv_sqrt[None, :]

class TrafficDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.stop = False

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            self.stop = True

def impute_data(df):
    return df.ffill().bfill().fillna(0)

def split_and_scale(df, scaler, train_frac=0.7, val_frac=0.1, test_frac=0.2):
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train + n_val]
    test_df = df.iloc[n_train + n_val:]

    print(f"Implementing a {train_frac:.0%}-{val_frac:.0%}-{test_frac:.0%} split")

    scaler.fit(train_df.values)

    train_scaled = scaler.transform(train_df.values)
    val_scaled = scaler.transform(val_df.values)
    test_scaled = scaler.transform(test_df.values)

    return {'train': train_scaled, 'val': val_scaled, 'test': test_scaled, 'scaler': scaler}

def load_graph(name, path, pkl_file):
    with open(os.path.join(path, pkl_file), 'rb') as f:
        graph_data = pickle.load(f, encoding='latin1')

    graph_dict = {
        'sensors': graph_data[0],
        'nodes': graph_data[1],
        'adj_matrix': torch.tensor(graph_data[2], dtype=torch.float32)
    }
    print(f'{name} graph loaded with {len(graph_dict['sensors'])} sensors')

    return graph_dict

def load_df(path, h5_file, key, agg_rule):
    df = pd.read_hdf(os.path.join(path, h5_file), key=key).astype('float32')
    df = df.resample(agg_rule).mean()
    print(f'Data aggregated into {agg_rule} intervals')
    
    if np.any(np.isnan(df)):
        print('Imputing missing values')
        df = impute_data(df)

    return df

def load_and_preprocess(name, kaggle_path, pkl_file, h5_file, key, agg_rule='5min', scaler=StandardScaler()):
    path = kagglehub.dataset_download(kaggle_path)

    graph_dict = load_graph(name, path, pkl_file)
    
    df = load_df(path, h5_file, key, agg_rule)
    data_dict = split_and_scale(df, scaler)

    return {'data': data_dict, 'graph': graph_dict}

def make_windows(data_array, lookback=12, horizon=3, step=1):
    x, y = [], []
    t = data_array.shape[0]
    
    for i in range(0, t - lookback - horizon + 1, step):
        x.append(data_array[i : i + lookback])
        y.append(data_array[i + lookback : i + lookback + horizon])
    
    return {'X': np.stack(x), 'y': np.stack(y)}

def flatten(model):
    if hasattr(model, 'flatten_parameters'):
        model.flatten_parameters()

    model.apply(lambda m: m.flatten_parameters() if hasattr(m, 'flatten_parameters') else None)

def train_step(model, loader, criterion, optimizer, device, clip_grads=True, max_norm=1.0):
    model.train()

    flatten(model)

    total_loss = 0.0

    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        
        preds = model(x_batch)
        loss = criterion(preds, y_batch)
        
        loss.backward()
        
        if clip_grads:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

        optimizer.step()
        
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    
    return avg_loss

def evaluate(criterion, loader, model, device, scaler=None):
    total_loss = 0.0 
    all_preds = []
    all_targets = []

    flatten(model)

    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                
            preds = model(x_batch)

            if isinstance(preds, tuple):
                preds = preds[0]

            loss = criterion(preds, y_batch)
            total_loss += loss.item()
            
            all_preds.append(preds.detach().cpu())
            all_targets.append(y_batch.detach().cpu())

    avg_loss = total_loss / len(loader)
    preds_inv, targets_inv = None, None

    if scaler and all_preds:
        preds_cat = torch.cat(all_preds, dim=0).numpy()
        targets_cat = torch.cat(all_targets, dim=0).numpy()

        preds_inv, targets_inv = inverse_transform_results(preds_cat, targets_cat, scaler)
        
    return avg_loss, preds_inv, targets_inv

def train_model(model_name, 
                epochs, 
                criterion, 
                optimizer, 
                model, 
                train_loader, 
                val_loader, 
                scheduler=None, 
                device='cpu',
                clip_grads=True,
                max_norm=1.0):
    
    history = {'train': [], 'val': []}

    print(f'Training the {model_name}')

    for epoch in range(epochs):
        train_loss = train_step(
            criterion=criterion,
            optimizer=optimizer,
            model=model,
            loader=train_loader, 
            device=device,
            clip_grads=clip_grads,
            max_norm=max_norm
        )

        eval_model = scheduler.swa_model if scheduler and scheduler.swa_active else model
           
        val_loss, _, _ = evaluate(
            criterion=criterion,
            loader=val_loader,
            model=eval_model,
            device=device
        )

        history['train'].append(train_loss)
        history['val'].append(val_loss)

        prefix = "SWA" if scheduler and scheduler.swa_active else "SGD"
        print(f"Epoch {epoch+1} [{prefix}] | Loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")

        if scheduler:
            scheduler.step(train_loader, val_loss)
            if scheduler.stop:
                break

    return history

def get_metrics(model_name, preds, y):
    mse = np.mean((preds - y) ** 2)
    mae = np.mean(np.abs(preds - y))
    wape = np.sum(np.abs(preds - y)) / np.sum(np.abs(y)) * 100
    
    return pd.DataFrame(
        data={model_name: [mse, mae, wape]},
        index = ['MSE', 'MAE', 'WAPE']
    )

def plot_training_history(name, model_name, history, loss_type, path=''):
    plt.figure(figsize=(8,5))
    plt.plot(history['train'], label='Train Loss', color='blue', linestyle='-')
    plt.plot(history['val'], label='Validation Loss', color='orange', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel(loss_type)
    plt.title(f'{model_name} Training & Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{path}/{name}_{model_name}_train_val_history.png',dpi=300,bbox_inches='tight')
    plt.show()

def plot_sample(name, sensor_idx, horizon_idx, preds, y, path=''):
    plt.figure(figsize=(12,4))
    plt.plot(y[:, horizon_idx, sensor_idx], label='Ground Truth', color='blue', linestyle='-', linewidth=1.2)
    plt.plot(preds[:, horizon_idx, sensor_idx], label='Prediction', color='orange', linestyle=':', linewidth=1.0)
    plt.xlabel('Sample')
    plt.ylabel('Traffic Speed')
    plt.title(f'{name}: Sensor {sensor_idx} - Horizon {horizon_idx+1}')
    plt.legend()
    plt.savefig(f'{path}/{name}_pred_vs_groundtruth.png',dpi=300,bbox_inches='tight')
    plt.show()

def inverse_transform_results(preds, targets, scaler):
    if scaler is None:
        return preds, targets

    s = preds.shape
    preds_2d = preds.reshape(-1, s[-1])
    targets_2d = targets.reshape(-1, s[-1])

    preds_inv = scaler.inverse_transform(preds_2d).reshape(s)
    targets_inv = scaler.inverse_transform(targets_2d).reshape(s)

    return preds_inv, targets_inv
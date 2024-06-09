import os
import cv2
import torch
import joblib
import timm
import torch.nn as nn
import numpy as np
from glob import glob
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torchmetrics.regression import R2Score
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import OneCycleLR
from torchvision.models import efficientnet
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
import sys
from typing import Any, cast, Dict, List, Optional, Union, Callable, OrderedDict


class CFG:
    verbose = 1  # Verbosity
    seed = 42  # Random seed
    # model_name = 'tf_efficientnetv2_b2'  # Name of pretrained classifier
    image_size = 224  # Input image size
    epochs = 32 # Training epochs
    batch_size = 48  # Batch size
    lr = 1e-3
    drop_remainder = True  # Drop incomplete batches
    num_classes = 6 # Number of classes in the dataset
    num_folds = 5 # Number of folds to split the dataset
    fold = 0 # Which fold to set as validation data
    class_names = ['X4_mean', 'X11_mean', 'X18_mean',
                   'X26_mean', 'X50_mean', 'X3112_mean',]
    aux_class_names = list(map(lambda x: x.replace("mean","sd"), class_names))
    num_classes = len(class_names)
    aux_num_classes = len(aux_class_names)
    precision = torch.float32
    
    
def build_augmenter():
    # Define Albumentations augmentations
    import random
    transform = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2),
        # A.HueSaturationValue(hue_shift_limit=0.1, sat_shift_limit=0.1, val_shift_limit=0.1, p=random.uniform(0, 0.1)),
        A.Cutout(num_holes=1, max_h_size=16, max_w_size=16, always_apply=False, p=random.uniform(0, 0.1)),
        A.HorizontalFlip(p=random.uniform(0, 0.2)),
        A.Rotate(limit=90, p=random.uniform(0, 0.2)),
        A.CenterCrop(height=CFG.image_size, width=CFG.image_size, always_apply=False, p=0.1),
        A.AdvancedBlur(blur_limit=(3,3), sigmaX_limit=(0.1,1.0), sigmaY_limit=(0.1,1.0), rotate_limit=(45), beta_limit=(0.5,8.0), noise_limit=(0.9,1.1), always_apply=False, p = 0.2),
        # A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, always_apply=False, p=0.2),
        A.Sharpen(alpha=(0.3,0.5), lightness=(0.5,1.0), always_apply=False, p=random.uniform(0, 0.1)),
        ToTensorV2(),
    ])

    return transform


def build_dataset(paths, features, labels=None, aux_labels=None, batch_size=32, cache=True, augment=True, repeat=True, shuffle=1024, cache_dir="", drop_remainder=False):
    from PlantDataset import PlantDataset
    dataset = PlantDataset(paths, features, labels, aux_labels, transform=build_augmenter(), augment=augment)

    if cache_dir != "" and cache:
        os.makedirs(cache_dir, exist_ok=True)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, drop_last=drop_remainder, pin_memory=True)

    return dataloader
    
def make_dataloader() -> DataLoader:
    PATH = '/Users/Shintaro/Documents/Kaggle-comps/PlantTraits/Data/planttraits2024'

    # Train + Valid
    df = pd.read_csv(f'{PATH}/train.csv')
    df['image_path'] = f'{PATH}/train_images/'+df['id'].astype(str)+'.jpeg'
    df.loc[:, CFG.aux_class_names] = df.loc[:, CFG.aux_class_names].fillna(-1)
    # display(df.head(2))

    # Test
    test_df = pd.read_csv(f'{PATH}/test.csv')
    test_df['image_path'] = f'{PATH}/test_images/'+test_df['id'].astype(str)+'.jpeg'
    FEATURE_COLS = test_df.columns[1:-1].tolist()
    # display(test_df.head(2))
    
    from sklearn.model_selection import StratifiedKFold

    # Assuming df is your dataframe containing file paths, features, labels, and fold information
    skf = StratifiedKFold(n_splits=CFG.num_folds, shuffle=True, random_state=42)

    # Create separate bin for each trait
    for i, trait in enumerate(CFG.class_names):
        bin_edges = np.percentile(df[trait], np.linspace(0, 100, CFG.num_folds + 1))
        df[f"bin_{i}"] = np.digitize(df[trait], bin_edges)

    df["final_bin"] = df[[f"bin_{i}" for i in range(len(CFG.class_names))]].astype(str).agg("".join, axis=1)

    df["fold"] = -1  # Initialize fold column

    # Perform the stratified split using final bin
    for fold, (train_idx, valid_idx) in enumerate(skf.split(df, df["final_bin"])):
        df.loc[valid_idx, "fold"] = fold
    
    # Sample from full data
    sample_df = df.copy()
    train_df = sample_df[sample_df.fold != CFG.fold]
    valid_df = sample_df[sample_df.fold == CFG.fold]
    print(f"# Num Train: {len(train_df)} | Num Valid: {len(valid_df)}")

    from sklearn.preprocessing import minmax_scale
    from sklearn.preprocessing import QuantileTransformer
    # Normalize features
    minmax = False
    if minmax == True:
        #get 0th and 100th percentile to use for the MinMaxScaler
        stds = [list(np.percentile(x.values, q=[0, 100])) for x in [pd.Series(df.loc[:, feat]) for feat in FEATURE_COLS]]
        # train_features = minmax_scale(X=train_df[FEATURE_COLS].values, feature_range=stds, axis=0)
        train_features = []
        valid_features = []
        for idx, (feat, std) in enumerate(zip(FEATURE_COLS, stds)):
            # print(feat)
            train_features.append(minmax_scale(X=train_df[feat].values, feature_range=(std[0], std[1])))
            valid_features.append(minmax_scale(X=valid_df[feat].values, feature_range=(std[0], std[1])))
        train_features = np.array(train_features).T
        valid_features = np.array(valid_features).T
        qt= QuantileTransformer(random_state=42)
        train_features = qt.fit_transform(train_features)
        valid_features = qt.fit_transform(valid_features)
    else:
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_df[FEATURE_COLS].values)
        valid_features = scaler.transform(valid_df[FEATURE_COLS].values)
        
        stds = [list(np.percentile(x.values, q=[1, 99])) for x in [pd.Series(df.loc[:, feat]) for feat in CFG.class_names+CFG.aux_class_names]]
        # print(stds)
        # for idx, (feat, std) in enumerate(zip(CFG.class_names, stds)):
        #     train_df[feat] = minmax_scale(X=train_df[feat].values, feature_range=(std[0],std[1]))
        #     valid_df[feat] = minmax_scale(X=valid_df[feat].values, feature_range=(std[0],std[1]))
        for _, (feat, (lower, upper)) in enumerate(zip(CFG.class_names+CFG.aux_class_names, stds)):
            # print(f'lower: {lower} | upper: {upper}')
            train_df[feat].clip(lower=lower, upper=upper, inplace=True)
            valid_df[feat].clip(lower=lower, upper=upper, inplace=True)
            
    # train_features = train_df[FEATURE_COLS].values
    # valid_features = valid_df[FEATURE_COLS].values
    
    # Extract file paths, features, labels, and fold information for train and validation sets
    train_paths = train_df.image_path.values
    train_labels = train_df[CFG.class_names].values
    train_aux_labels = train_df[CFG.aux_class_names].values

    valid_paths = valid_df.image_path.values
    valid_labels = valid_df[CFG.class_names].values
    valid_aux_labels = valid_df[CFG.aux_class_names].values

    # Build datasets
    train_dataloader = build_dataset(train_paths, train_features, train_labels, train_aux_labels,
                            batch_size=CFG.batch_size,
                            repeat=True, shuffle=True, augment=True, cache=False)
    valid_dataloader = build_dataset(valid_paths, valid_features, valid_labels, valid_aux_labels,
                            batch_size=CFG.batch_size,
                            repeat=False, shuffle=False, augment=False, cache=False)
    
    return train_dataloader, valid_dataloader

class R2Loss(nn.Module):
    def __init__(self, use_mask=False):
        super(R2Loss, self).__init__()
        self.use_mask = use_mask

    def forward(self, y_pred, y_true, std:Optional[float] = None):
        if self.use_mask:
            mask = (y_true != -1)
            y_true = torch.where(mask, y_true, torch.zeros_like(y_true))
            y_pred = torch.where(mask, y_pred, torch.zeros_like(y_pred))

        # mask2 = (y_true-y_pred <= std)
        # print(f'mask: {mask2}')
        # masked = torch.where(mask2, y_pred, torch.zeros_like(y_pred))
        # print(f"masked: {masked}")
        SS_res = torch.sum((y_true - y_pred)**2, dim=0)  # (B, C) -> (C,)
        # print(f'ss_res: {SS_res}')
        SS_tot = torch.sum((y_true - torch.mean(y_true, dim=0))**2, dim=0)  # (B, C) -> (C,)
        r2_loss = SS_res / (SS_tot + 1e-6)  # (C,)
        return torch.mean(r2_loss)  # ()
    
def create_model(device:str) -> nn.Module:
    from VitLike import CNNLIN
    import ModelCFGs as cfgs
    cfg = cfgs.CNNLIN_cfg
    # print(cfg.cnn_cfg)
    # model =  CNNLIN(cfg.cnn_cfg).to(device)
    # model = toy_model()
    
    
    from Inception_CFGs import Incept_CFGS
    from Inception_Model_Builder import Inception_classifier
    config = Incept_CFGS.Incept_CNN_CFG
    model = Inception_classifier(inception_cfg=     config["Incept_v1"]["incept_cfg"], 
                                 cnn_classifier_cfg=config["Incept_v1"]["cnn_classifier_cfg"], 
                                 aux_classifer_cfg= config["Incept_v1"]["aux_classifier_cfg"], 
                                 classifier_cfg=    config["Incept_v1"]["classifier_cfg"])
    
    model.to(device)
    count_parameters(model)
    return model

def toy_model()-> nn.Module:
    from cube_cnn import NLWCNN, BestModel
    # model = nn.Conv2d(4, 64, kernel_size=5, padding='same')
    # model = NLWCNN().to(device='cuda')
    model = BestModel().to(device='cuda')
    return model
    
def train(model:nn.Module, train_dataloader:DataLoader, valid_dataloader:DataLoader, device:str):
    criterion_img = R2Loss(use_mask=False)
    criterion_aux = R2Loss(use_mask=False)
    criterion_c = nn.BCELoss()

    best_model_path = '/Users/Shintaro/Documents/Kaggle-comps/PlantTraits/best_model.pth'
    best_r2_score = -float('inf')
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=CFG.lr)

    from torchmetrics.regression import R2Score
    from tqdm import tqdm

    load_model = False
    load_model_path = '/Users/Shintaro/Documents/Kaggle-comps/PlantTraits/best_models/Vit-J-standard_r2_0.09.pth'
    # load_model_path = best_model_path
    if load_model:
        model.load_state_dict(torch.load(best_model_path))
        print(f'{best_model_path} Model loaded')
        
    metric = R2Score(num_outputs=6, multioutput='uniform_average').to(device)
    print(count_parameters(model))
    # train_dataloader, valid_dataloader = build_dataloader()

    losses = []
    R2s = []
    for epoch in range(CFG.epochs):
        model.train()
        total_train_r2 = 0.0
        train_batches = 0
        total_train_loss = 0.0
        for batch_idx, (inputs_dict, (targets, aux_targets, c_label)) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            
            input_img = inputs_dict['images'].to(device, dtype=torch.float32)
            input_feat = inputs_dict['features'].to(device, dtype=torch.float32)
            targets = targets.to(device, dtype=torch.float32)
            aux_targets = aux_targets.to(device, dtype=torch.float32)
            c_label = c_label.to(device, dtype=torch.float32)
            # with torch.autocast(device_type="cuda"):
            head_out, c_out = model(input_img, input_feat)
            loss_c = criterion_c(c_out, c_label)
            # loss_head = criterion_img(head_out, targets)
            loss_head_lower = criterion_img(head_out - aux_targets, targets)
            loss_head_upper = criterion_img(head_out + aux_targets, targets)
            loss_head_final = loss_head_lower if loss_head_lower < loss_head_upper else loss_head_upper
            loss = (loss_head_final+loss_c)
            
            loss.backward()
            optimizer.step()
            
            r2_value = metric(head_out, targets)
            if batch_idx % 500 == 0:
                print(f'cnn loss: {loss_c.item()} | upper: {loss_head_lower.item()} | upper: {loss_head_upper.item()} | final: {loss_head_final.item()} | r2: {r2_value}')

            
            
            total_train_loss += loss.item()
            total_train_r2 += r2_value.item()
            
            train_batches += 1
            
        avg_train_r2 = total_train_r2 / train_batches
        avg_train_loss = total_train_loss / train_batches
        losses.append(avg_train_loss)
        R2s.append(avg_train_r2)
        # avg_train_just_aux = train_just_aux / train_batches
        print(f"Epoch: {epoch+1}/{CFG.epochs}, Average Train R2 Score: {avg_train_r2}, Average Train Loss: {avg_train_loss}")
        # print(losses)
        # print(sum(losses))
        model.eval()
        
        total_val_r2 = 0.0
        val_batches = 0
        total_val_loss = 0.0
        
        with torch.no_grad():
            for val_batch_idx, (val_inputs_dict, (val_targets, val_aux_targets, c_label)) in enumerate(tqdm(valid_dataloader)):
                val_inputs_img = val_inputs_dict['images'].to(device, dtype=CFG.precision)
                val_input_feat = val_inputs_dict['features'].to(device, dtype=CFG.precision)
                val_targets = val_targets.to(device, dtype=CFG.precision)
                val_aux_targets = val_aux_targets.to(device, dtype=CFG.precision)
                val_c_label = c_label.to(device, dtype=CFG.precision)
                # val_all_target = torch.concat([val_targets, val_aux_targets], dim=1)
                
                val_head_out, val_c_out= model(val_inputs_img, val_input_feat)
                val_loss_c = criterion_c(val_c_out, val_c_label)
                
                val_loss_upper = criterion_img(val_head_out+val_aux_targets, val_targets)
                val_loss_lower = criterion_img(val_head_out-val_aux_targets, val_targets)
                val_loss_final = val_loss_lower if val_loss_lower < val_loss_upper else val_loss_upper
                
                
                total_val_loss += (val_loss_c.item() + val_loss_final.item())
                
                
                r2_value = metric(val_head_out, val_targets)
                
                total_val_r2 += r2_value.item()
                val_batches += 1
                
        avg_val_r2 = total_val_r2 / val_batches
        avg_val_loss = total_val_loss / val_batches
        print(f"Epoch: {epoch + 1}/{CFG.epochs}, Average Val R2 Score: {avg_val_r2}, Average Val Loss: {avg_val_loss}")
        
        if avg_val_r2 > best_r2_score:
            best_r2_score = avg_val_r2
            torch.save(model.state_dict(), best_model_path)
        # if epoch% 3 == 0:
        #     train_dataloader, valid_dataloader = build_dataloader()
    torch.save(model.state_dict(), '/Users/Shintaro/Documents/Kaggle-comps/PlantTraits/best_models/overnight.pth')
        

def train_test(model:nn.Module, train_dataloader:DataLoader, valid_dataloader:DataLoader, device:str):
    criterion_img = R2Loss(use_mask=False)
    criterion_aux = R2Loss(use_mask=False)
    criterion_c = nn.BCELoss()

    best_model_path = '/Users/Shintaro/Documents/Kaggle-comps/PlantTraits/best_model.pth'
    best_r2_score = -float('inf')
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', factor=0.1, threshold=1e-4, patience=3, cooldown=4, verbose=True)
    # optimizer = torch.optim.SGD(model.parameters(), lr=CFG.lr)

    from torchmetrics.regression import R2Score
    from tqdm import tqdm

    load_model = False
    load_model_path = '/Users/Shintaro/Documents/Kaggle-comps/PlantTraits/best_models/nonlin_0.4.pth'
    # load_model_path = best_model_path
    if load_model:
        model.load_state_dict(torch.load(best_model_path))
        print(f'{best_model_path} Model loaded')
        
    metric = R2Score(num_outputs=3, multioutput='uniform_average').to(device)
    print(count_parameters(model))
    # train_dataloader, valid_dataloader = build_dataloader()

    losses = []
    R2s = []
    for epoch in range(CFG.epochs):
        model.train()
        total_train_r2 = 0.0
        train_batches = 0
        total_train_loss = 0.0
        for batch_idx, (inputs_dict, (targets, aux_targets, c_label)) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            
            input_img = inputs_dict['images'].to(device, dtype=torch.float32)
            features = inputs_dict['features'].to(device, dtype=torch.float32)
            
            targets = targets.to(device, dtype=torch.float32)
            aux_targets = aux_targets.to(device, dtype=torch.float32)
            # torch.autograd.set_detect_anomaly(True)
            head_out_1, head_out_2= model(input_img, features)
            loss_1 = criterion_img(head_out_1, targets[:,0::2], aux_targets)
            loss_2 = criterion_img(head_out_2, targets[:,1::2], aux_targets)
            loss = loss_1+loss_2
            loss.backward()
            # break
            
            r2_value_1 = metric(head_out_1, targets[:,0::2])
            r2_value_2 = metric(head_out_2, targets[:,1::2])
            r2_value = r2_value_1+r2_value_2
            # print(r2_value.item())
            # scheduler.step(loss)
            optimizer.step()
            

            total_train_loss += loss.item()
            total_train_r2 += r2_value.item()
            
            train_batches += 1
            
        avg_train_r2 = total_train_r2 / train_batches
        avg_train_loss = total_train_loss / train_batches
        # scheduler.step(avg_train_loss)
        losses.append(avg_train_loss)
        R2s.append(avg_train_r2)
        # avg_train_just_aux = train_just_aux / train_batches
        print(f"Epoch: {epoch+1}/{CFG.epochs}, Average Train R2 Score: {avg_train_r2}, Average Train Loss: {avg_train_loss}")
        # print(losses)
        # print(sum(losses))
        model.eval()
        
        total_val_r2 = 0.0
        val_batches = 0
        total_val_loss = 0.0
        
        with torch.no_grad():
            for val_batch_idx, (val_inputs_dict, (val_targets, val_aux_targets, c_label)) in enumerate(tqdm(valid_dataloader)):
                val_inputs_img = val_inputs_dict['images'].to(device, dtype=CFG.precision)
                val_input_feat = val_inputs_dict['features'].to(device, dtype=CFG.precision)
                val_targets = val_targets.to(device, dtype=CFG.precision)
                val_aux_targets = val_aux_targets.to(device, dtype=CFG.precision)
                # val_c_label = c_label.to(device, dtype=CFG.precision)
                # val_all_target = torch.concat([val_targets, val_aux_targets], dim=1)
                
                val_head_out_1, val_head_out_2 = model(val_inputs_img, val_input_feat)
                val_loss_1 = criterion_img(val_head_out_1, val_targets[:,0::2], val_aux_targets)
                val_loss_2 = criterion_img(val_head_out_2, val_targets[:,1::2], val_aux_targets)
                val_loss = val_loss_1+val_loss_2
                total_val_loss += val_loss
                
                
                r2_value_1 = metric(val_head_out_1, val_targets[:,0::2])
                r2_value_2 = metric(val_head_out_2, val_targets[:,1::2])
                r2_value = r2_value_1+r2_value_2
                
                total_val_r2 += r2_value.item()
                val_batches += 1
                
        avg_val_r2 = total_val_r2 / val_batches
        avg_val_loss = total_val_loss / val_batches
        print(f"Epoch: {epoch + 1}/{CFG.epochs}, Average Val R2 Score: {avg_val_r2}, Average Val Loss: {avg_val_loss}")
        
                
        if avg_val_r2 > best_r2_score:
            best_r2_score = avg_val_r2
            torch.save(model.state_dict(), best_model_path)
        
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data)
    
    torch.save(model.state_dict(), '/Users/Shintaro/Documents/Kaggle-comps/PlantTraits/best_models/overnight.pth')

def train_test_2(model:nn.Module, train_dataloader:DataLoader, valid_dataloader:DataLoader, device:str):
    criterion_img = R2Loss(use_mask=False)
    criterion_aux = nn.MSELoss()
    criterion_c = nn.BCELoss()

    best_model_path = '/Users/Shintaro/Documents/Kaggle-comps/PlantTraits/best_model.pth'
    best_r2_score = -float('inf')
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', factor=0.1, threshold=1e-4, patience=3, cooldown=4, verbose=True)
    # optimizer = torch.optim.SGD(model.parameters(), lr=CFG.lr)

    from torchmetrics.regression import R2Score
    from tqdm import tqdm

    load_model = False
    load_model_path = '/Users/Shintaro/Documents/Kaggle-comps/PlantTraits/best_models/nonlin_0.4.pth'
    # load_model_path = best_model_path
    if load_model:
        model.load_state_dict(torch.load(best_model_path))
        print(f'{best_model_path} Model loaded')
        
    metric = R2Score(num_outputs=6, multioutput='uniform_average').to(device)
    print(count_parameters(model))
    # train_dataloader, valid_dataloader = build_dataloader()

    losses = []
    R2s = []
    for epoch in range(CFG.epochs):
        model.train()
        total_train_r2 = 0.0
        train_batches = 0
        total_train_loss = 0.0
        for batch_idx, (inputs_dict, (targets, aux_targets, c_label)) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            
            input_img = inputs_dict['images'].to(device, dtype=torch.float32)
            features = inputs_dict['features'].to(device, dtype=torch.float32)
            
            targets = targets.to(device, dtype=torch.float32)
            aux_targets = aux_targets.to(device, dtype=torch.float32)
            # torch.autograd.set_detect_anomaly(True)
            head_out = model(input_img, features)
            loss_head = criterion_img(head_out, targets)
            # loss_aux = criterion_aux(aux_out, aux_targets)
            loss = loss_head 
            loss.backward()
            # break
            
            r2_value = metric(head_out, targets)
            # print(r2_value.item())
            # scheduler.step(loss)
            optimizer.step()
            

            total_train_loss += loss.item()
            total_train_r2 += r2_value.item()
            
            train_batches += 1
            
        avg_train_r2 = total_train_r2 / train_batches
        avg_train_loss = total_train_loss / train_batches
        # scheduler.step(avg_train_loss)
        losses.append(avg_train_loss)
        R2s.append(avg_train_r2)
        # avg_train_just_aux = train_just_aux / train_batches
        print(f"Epoch: {epoch+1}/{CFG.epochs}, Average Train R2 Score: {avg_train_r2}, Average Train Loss: {avg_train_loss}")
        # print(losses)
        # print(sum(losses))
        model.eval()
        
        total_val_r2 = 0.0
        val_batches = 0
        total_val_loss = 0.0
        
        with torch.no_grad():
            for val_batch_idx, (val_inputs_dict, (val_targets, val_aux_targets, c_label)) in enumerate(tqdm(valid_dataloader)):
                val_inputs_img = val_inputs_dict['images'].to(device, dtype=CFG.precision)
                val_input_feat = val_inputs_dict['features'].to(device, dtype=CFG.precision)
                val_targets = val_targets.to(device, dtype=CFG.precision)
                val_aux_targets = val_aux_targets.to(device, dtype=CFG.precision)
                # val_c_label = c_label.to(device, dtype=CFG.precision)
                # val_all_target = torch.concat([val_targets, val_aux_targets], dim=1)
                
                val_head_out = model(val_inputs_img, val_input_feat)
                val_head_loss = criterion_img(val_head_out, val_targets)
                # val_aux_loss = criterion_aux(val_aux_out, val_aux_targets)
                val_loss = val_head_loss
                
                total_val_loss += val_loss
                
                
                r2_value = metric(val_head_out, val_targets)
                
                total_val_r2 += r2_value.item()
                val_batches += 1
        avg_val_r2 = total_val_r2 / val_batches
        avg_val_loss = total_val_loss / val_batches
        print(f"Epoch: {epoch + 1}/{CFG.epochs}, Average Val R2 Score: {avg_val_r2}, Average Val Loss: {avg_val_loss}")
        
                
        if avg_val_r2 > best_r2_score:
            best_r2_score = avg_val_r2
            torch.save(model.state_dict(), best_model_path)
        
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data)
    
    torch.save(model.state_dict(), '/Users/Shintaro/Documents/Kaggle-comps/PlantTraits/best_models/overnight.pth')

def get_test_results(weight_name:str, model:nn.Module, device:str = 'cuda'):
    best_model_path = '/Users/Shintaro/Documents/Kaggle-comps/PlantTraits/best_models/'+weight_name
    model.load_state_dict(torch.load(best_model_path))

    PATH = '/Users/Shintaro/Documents/Kaggle-comps/PlantTraits/Data/planttraits2024'
    # Test
    test_df = pd.read_csv(f'{PATH}/test4/test.csv')
    test_df['image_path'] = f'{PATH}/test2_images/'+test_df['id'].astype(str)+'.jpeg'
    FEATURE_COLS = test_df.columns[1:-1].tolist()
    # display(test_df.head(2))
    
    scaler = StandardScaler()
    # Test
    test_paths = test_df.image_path.values
    test_features = scaler.fit_transform(test_df[FEATURE_COLS].values) 
    test_ds = build_dataset(test_paths, test_features, batch_size=CFG.batch_size,
                            repeat=False, shuffle=False, augment=False, cache=False)

    model.eval()  # Set the model to evaluation mode

    # List to store predictions
    all_predictions = []

    # Iterate over batches in the test data loader
    for batch_idx, inputs_dict in enumerate(tqdm(test_ds, desc='Testing')):
        # Extract images and features from the inputs_dict
        inputs_images = inputs_dict['images'].to(device, dtype=CFG.precision)  # Assuming 'device' is the target device
        inputs_features = inputs_dict['features'].to(device, dtype=CFG.precision)

        # Forward pass
        with torch.no_grad():
            outputs = model(inputs_images, inputs_features)

        # Get predictions
        predictions = outputs.cpu().numpy()  # Assuming 'head' is the main task output
        
        # Append predictions to the list
        all_predictions.append(predictions)

    # Concatenate predictions for all batches
    all_predictions = np.concatenate(all_predictions, axis=0)
    print(all_predictions)
    pred_df = test_df[["id"]].copy()
    target_cols = [x.replace("_mean","") for x in CFG.class_names]
    print(pred_df)
    pred_df[target_cols] = all_predictions.tolist()
    # cols = ["id","X4","X11","X18","X50","X26","X3112"]
    # pred_df = pred_df[cols]
    print(pred_df)
    pred_df.to_csv("submissionNLL0.4.csv", index=False)
    # print(pred_df)
    # sub_df = pd.read_csv(f'{PATH}/sample_submission.csv')
    # sub_df = sub_df[["id"]].copy()
    # sub_df = sub_df.merge(pred_df, on="id", how="left")
    # sub_df.to_csv("submissionNLL0.4.csv", index=False)

def main():
    from ConvNext_Model_Builder import ConvNext, InceptionV4, InceptionResNetV2, Double_InceptionV4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # train_dataloader, valid_dataloader = make_dataloader()
    # model = Double_InceptionV4(6).to(device=device)
    # model = create_model(device)
    model = toy_model()
    
    # print(model)
    # train(model, train_dataloader, valid_dataloader, device)
    # test(train_dataloader, model=create_model(device))
    # train_test(model, train_dataloader, valid_dataloader, device)
    # train_test_2(model, train_dataloader, valid_dataloader, device)

    get_test_results("nonlin_0.4_original.pth", model, device)
    
# Counts number of trainable parameters in Model
# From https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/7
# By: Bladassarre.fe and vsmolyakov
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test(train_dataloader:Optional[DataLoader]=None, model:Optional[nn.Module]=None):
    # model = torch.load('/Users/Shintaro/Documents/Kaggle-comps/PlantTraits/best_models/nonlin_0.4.pth')
    # modelarch = list(model.keys())
    # for x in modelarch:
    #     print(x)
    # c = torch.zeros([32,4])
    # a = [0,2,4]
    # b = [1,3,5]
    # a = torch.tensor(a)
    # b = torch.tensor(b)
    # c = torch.zeros(6)
    # c[0::2] = a
    # c[1::2] = b
    # c = torch.tensor([torch.cat([a[x], b[x]]) for x in range(len(a))])
    # c = torch.tensor(c)
    # print(c.shape[0])
    # from Inception_CFGs import Incept_CFGS
    # from Inception_Model_Builder import Inception_classifier
    # config = Incept_CFGS.Incept_CNN_CFG
    # model = Inception_classifier(inception_cfg=config["Incept_v1"]["incept_cfg"], classifier_cfg=config["Incept_v1"]["classifier_cfg"])
    # print(model)
    # inps, tars = next(iter(train_dataloader))
    # imgs = inps["images"]
    # # print(imgs)
    # print(imgs.shape)
    # out = model(imgs.to(device="cuda", dtype=torch.float32))
    # print(model.weight.shape)
    # # print(model.weight)
    # print(out.shape)
    # model = create_model("cuda")
    # load_model = False

    # load_model_path = '/Users/Shintaro/Documents/Kaggle-comps/PlantTraits/best_models/Vit-J-standard_r2_0.09.pth'
    # # load_model_path = best_model_path
    # best_model_path = '/Users/Shintaro/Documents/Kaggle-comps/PlantTraits/best_model.pth'

    # if load_model:
    #     model.load_state_dict(torch.load(best_model_path))
    #     print(f'{best_model_path} Model loaded')
    # print(model)
    # print(count_parameters(model))
    
    import B_spline as B
    num_spline = 5
    num_sample = 100
    num_grid_interval = 10
    k = 3
    x = torch.normal(0,1,size=(num_spline, num_sample))
    grids = torch.einsum('i,j->ij', torch.ones(num_spline,), torch.linspace(-1,1,steps=num_grid_interval+1))
    b_spline_out = B.B_batch(x, grids, k=k)
    print(b_spline_out)
    
        
if __name__ == '__main__':
    sys.exit(main())
    # sys.exit(test())
    

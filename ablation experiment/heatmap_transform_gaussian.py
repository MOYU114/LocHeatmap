import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import os
import utils
import utils_model
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
map_size=128
input_dim = 2
threshold = 0.05
lr = 0.00001
Rh = 30

MODEL_PATH = "model/PDF_model_"

print("read data")
training_epochs = 500
change_epoch = 20
batch_size = 1
dataset_PATH = "./data/test_data/merge.pth"
S_PATH = "./data/test_data/S_coord_fixed.csv"
training_end = 54

dataset = torch.load(dataset_PATH)
gen_map = dataset[:training_end, 0, :, :].type(torch.FloatTensor).unsqueeze(1).to(device)
RSS_real = dataset[:training_end, 1, :, :].type(torch.FloatTensor).unsqueeze(1).to(device)
AOA_real = dataset[:training_end, -1, :, :].type(torch.FloatTensor).unsqueeze(1).to(device)
S_list = pd.read_csv(S_PATH, header=None)
S_list = torch.from_numpy(S_list.to_numpy())
S_list = S_list.type(torch.FloatTensor).to(device)
S_list = S_list[:training_end]
padding = (0, 128 - 117, 0, 128 - 117)  # (left, right, top, bottom)
_map = F.pad(gen_map, padding, mode='constant', value=0)
RSS_input = F.pad(RSS_real, padding, mode='constant',
                     value=torch.min(RSS_real.unsqueeze(0)).float())
AOA_input = F.pad(AOA_real, padding, mode='constant', value=0)

S_map = torch.zeros((len(RSS_input),1,map_size,map_size),device=device)
PDF_input = torch.cat((RSS_input,AOA_input, _map), dim=1)
# PDF_real = PDF
PDF_real = utils.generate_heatmap(S_list.int(), img_size=map_size, func="gaussian")

print("read data")

utils.draw_map(PDF_real[0], f"PDF_real_norm")

#dataset_aoa = [rss_input, AOA_real]
training_dataloader = DataLoader(TensorDataset(PDF_input[0:int(len(PDF_input) // batch_size) * batch_size],
                                               PDF_real[0:int(len(PDF_real) // batch_size) * batch_size]),
                                 batch_size=batch_size, shuffle=True)
dsnt = utils_model.DSNT(map_size, map_size)

rss_model = utils_model.SimpleAutoencoder(2).to(device)
rss_model.load_state_dict(torch.load("model/rss_model__test_250epoch.pth", weights_only=True))
aoa_model = utils_model.SimpleAutoencoder(2).to(device)
aoa_model.load_state_dict(torch.load("model/aoa_model__test_250epoch.pth", weights_only=True))

model = utils_model.LocUNet(input_dim).to(device)
# model = utils_model.SimpleAutoencoder(input_dim).to(device)
optimizer =torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

for epoch in range(training_epochs):
    epoch_loss = 0
    with tqdm(total=len(training_dataloader)) as t:

        for pdf_input, PDF_real in training_dataloader:

            rss_training = pdf_input[:, 0, :, :].unsqueeze(1)
            aoa_training = pdf_input[:, 1, :, :].unsqueeze(1)
            _map_training = pdf_input[:, 1, :, :].unsqueeze(1)

            mask = torch.rand(batch_size, 1, 128, 128) > (1-threshold)
            mask = mask.to(device)
            rss_uav = utils.get_rand_sparse_matrix_norm_from_median(rss_training, batch_size, mask, max_ctl="rss")
            _map_training = utils.get_norm_for_height(_map_training, batch_size)
            rss_uav_sparse = torch.cat((rss_uav, _map_training), dim=1)

            aoa_uav = utils.get_rand_sparse_matrix_norm(aoa_training, batch_size, mask, max_ctl="aoa")
            _map_training = utils.get_norm_for_height(_map_training, batch_size)
            aoa_uav_sparse = torch.cat((aoa_uav, _map_training), dim=1)

            aoa_pred = aoa_model(aoa_uav_sparse)
            rss_pred = rss_model(rss_uav_sparse)
            aoa_pred_norm = utils.get_norm(aoa_pred, batch_size)
            rss_pred_norm = utils.get_norm(rss_pred, batch_size)

            PDF_real_norm = utils.get_norm(PDF_real, batch_size)
            # aoa_pred_norm_input = torch.cat((aoa_pred_norm, _map_training), dim=1)
            optimizer.zero_grad()
            pred_norm = torch.cat((rss_pred_norm, aoa_pred_norm), dim=1)
            PDF_pred = model(pred_norm)
            PDF_pred = utils.get_norm(PDF_pred, batch_size)
            # S_pred = torch.tensor(utils.find_maxarg_single(PDF_pred),dtype=torch.float32)
            S_pred = dsnt((PDF_pred/torch.sum(PDF_pred))).squeeze()
            S_real = torch.tensor(utils.find_maxarg_single(PDF_real_norm),dtype=torch.float32,device=device)
            loss = 0.8*criterion(PDF_pred,PDF_real_norm) + 0.2*criterion(S_pred,S_real)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss / len(PDF_pred)
            t.set_postfix(loss=loss / len(PDF_pred))
            t.update(1)
        t.close()
    print(f"Training ev model: Epoch {epoch}, loss: {epoch_loss / len(training_dataloader)}")
    torch.set_printoptions(profile="full")
    if (epoch+1)%change_epoch ==0:
        utils.draw_map(rss_pred_norm[0], f"rss_pred_norm")
        utils.draw_map(aoa_pred_norm[0], f"aoa_pred_norm")
        utils.draw_map(PDF_pred[0], f"PDF_pred")
        utils.draw_map(PDF_real_norm[0], f"PDF_real_norm")
        S_pred = torch.tensor(utils.find_maxarg_single(PDF_pred), dtype=torch.float32)
        S_real = torch.tensor(utils.find_maxarg_single(PDF_real_norm), dtype=torch.float32)
        print(S_pred)
        print(S_real)
        threshold += 0.01

# torch.save(model.state_dict(), MODEL_PATH + "_test_log.pth")
torch.save(model.state_dict(),MODEL_PATH+ "_test_gaussian.pth")
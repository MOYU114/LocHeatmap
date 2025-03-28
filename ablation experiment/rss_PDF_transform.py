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
input_dim = 1
threshold = 0.05
lr = 0.00001
Rh = 30
MODEL_PATH = "./model/rss_PDF_model_"

def gen_norm_matrix(M, N, origin_x, origin_y):
    alpha0, beta0, = -5, -5
    x = torch.arange(M, device=device).view(M, 1).expand(M, N)
    y = torch.arange(N, device=device).view(1, N).expand(M, N)
    s_x = torch.full((M, N), origin_x, device=device)
    s_y = torch.full((M, N), origin_y, device=device)
    dist = torch.sqrt(torch.pow((s_x - x), 2) + torch.pow((s_y - y), 2))
    rss = beta0 + alpha0 * torch.log10(dist)
    rss[origin_x,origin_y] = 0
    return rss

print("read data")

training_epochs = 1000
change_epoch = 40
batch_size = 1
dataset_PATH = "./data/test_data/merge.pth"
S_PATH = "./data/test_data/S_coord_fixed.csv"
MODELRuf_PATH = "model/S_Ruf_3layers_rss_aoa_norm_3inputs_20.pth"
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
rss_PDF = torch.zeros((len(RSS_input),1,map_size,map_size),device=device)

for i in range(len(rss_PDF)):
    sx= S_list[i,0].int()
    sy =S_list[i, 1].int()
    temp = gen_norm_matrix(map_size, map_size, sx, sy)
    rss_PDF[i,0] = temp
rss_input = torch.cat((RSS_input, _map), dim=1)
RSS_real = rss_PDF

print("read data")


#dataset_aoa = [rss_input, AOA_real]
training_dataloader = DataLoader(TensorDataset(rss_input[0:int(len(rss_input) // batch_size) * batch_size],
                                               RSS_real[0:int(len(RSS_real) // batch_size) * batch_size]),
                                 batch_size=batch_size, shuffle=True)


rss_model = utils_model.SimpleAutoencoder(2).to(device)
rss_model.load_state_dict(torch.load("./model/rss_model__test_250epoch.pth", weights_only=True))

model = utils_model.UNet_rss(input_dim).to(device)
# model = utils_model.SimpleAutoencoder(input_dim).to(device)
optimizer =torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

for epoch in range(training_epochs):
    epoch_loss = 0
    with tqdm(total=len(training_dataloader)) as t:

        for rss_input, rss_PDF_real in training_dataloader:

            rss_training = rss_input[:, 0, :, :].unsqueeze(1)
            _map_training = rss_input[:, 1, :, :].unsqueeze(1)
           
            mask = torch.rand(batch_size, 1, 128, 128) > (1-threshold)
            mask = mask.to(device)
            rss_uav = utils.get_rand_sparse_matrix_norm_from_median(rss_training, batch_size, mask, max_ctl="rss")
            _map_training = utils.get_norm_for_height(_map_training, batch_size)
            rss_uav_sparse = torch.cat((rss_uav, _map_training), dim=1)
            rss_pred = rss_model(rss_uav_sparse)

            rss_pred_norm = utils.get_norm(rss_pred, batch_size)
            rss_PDF_real_norm = utils.get_norm(rss_PDF_real, batch_size)
            # aoa_pred_norm_input = torch.cat((aoa_pred_norm, _map_training), dim=1)
            optimizer.zero_grad()
            rss_PDF_pred = model(rss_pred_norm)

            loss = criterion(rss_PDF_pred,rss_PDF_real_norm)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss / len(rss_PDF_pred)
            t.set_postfix(loss=loss / len(rss_PDF_pred))
            t.update(1)
        t.close()
    print(f"Training ev model: Epoch {epoch}, loss: {epoch_loss / len(training_dataloader)}")
    torch.set_printoptions(profile="full")
    if (epoch+1)%change_epoch ==0:
        utils.draw_map(rss_pred_norm[0],f"rss_pred_norm")
        utils.draw_map(rss_PDF_real[0], f"rss_PDF_real")
        utils.draw_map(rss_PDF_pred[0], f"rss_PDF_pred")
        threshold += 0.01

torch.save(model.state_dict(),MODEL_PATH+"real.pth")
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
map_size = 128
threshold = 0.05
input_dim = 2
lr = 0.00001
Rh = 30
MODEL_PATH = "./model/aoa_model_"

print("read data")
training_epochs = 250
change_epoch = 10
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

S_map = torch.zeros((len(AOA_input), 1, map_size, map_size), device=device)

for i in range(len(S_map)):
    sx = S_list[i, 0].int()
    sy = S_list[i, 1].int()
    S_map[i, 0, sx, sy] = 1
aoa_input = torch.cat((AOA_input, _map, S_map), dim=1)
AOA_real = AOA_input

print("read data")

training_dataloader = DataLoader(TensorDataset(aoa_input[0:int(len(aoa_input) // batch_size) * batch_size],
                                               AOA_real[0:int(len(AOA_real) // batch_size) * batch_size]),
                                 batch_size=batch_size, shuffle=True)

model = utils_model.SimpleAutoencoder(input_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

for epoch in range(training_epochs):
    epoch_loss = 0
    with tqdm(total=len(training_dataloader)) as t:

        for aoa_input, aoa_real in training_dataloader:
            aoa_training = aoa_input[:, 0, :, :].unsqueeze(1)
            _map_training = aoa_input[:, 1, :, :].unsqueeze(1)
            S_map_training = aoa_input[:, 2, :, :].unsqueeze(1)

            mask = torch.rand(batch_size, 1, 128, 128) > (1 - threshold)
            mask = mask.to(device)

            aoa_uav = utils.get_rand_sparse_matrix_norm(aoa_training, batch_size, mask, max_ctl="aoa")
            aoa_real_norm = utils.get_norm(aoa_real, batch_size, max_ctl="aoa")
            _map_training = utils.get_norm_for_height(_map_training, batch_size)
            # aoa_input_mask = torch.cat((aoa_uav, _map_training, S_map_training), dim=1)
            aoa_input_mask = torch.cat((aoa_uav, _map_training), dim=1)

            optimizer.zero_grad()
            aoa_pred = model(aoa_input_mask)
            s_list = utils.find_maxarg_multi(S_map_training, 0)

            loss = criterion(aoa_pred, aoa_real_norm)
            loss.backward()
            optimizer.step()

            epoch_loss += loss / len(aoa_pred)
            t.set_postfix(loss=loss / len(aoa_pred))
            t.update(1)
        t.close()
    print(f"Training ev model: Epoch {epoch}, loss: {epoch_loss / len(training_dataloader)}")
    torch.set_printoptions(profile="full")

    if (epoch + 1) % change_epoch == 0:
        print(f"current threshold is {threshold}")
        utils.draw_map(aoa_uav[0],f"aoa_uav")
        utils.draw_map(aoa_pred[0], f"aoa_pred")
        utils.draw_map(aoa_real_norm[0], f"aoa_real_norm")
        threshold += 0.01
torch.save(model.state_dict(), MODEL_PATH + "_test_250epoch.pth")
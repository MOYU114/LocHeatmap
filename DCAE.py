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

training_model_list = ["DCAE", "DCAE_simple"]
model_index = 0
map_size = 128
threshold = 0.05
lr = 0.00001
Rh = 30

MODEL_PATH = f"./model/DCAE_model_{training_model_list[model_index]}.pth"

training_epochs = 1000
change_epoch = 40
batch_size = 1

print(f"training: "+MODEL_PATH)

print("read data")
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
ori_map_size = gen_map.size(2)
padding = (0, map_size - ori_map_size, 0, map_size - ori_map_size)  # (left, right, top, bottom)
_map = F.pad(gen_map, padding, mode='constant', value=0)
RSS_input = F.pad(RSS_real, padding, mode='constant',
                  value=torch.min(RSS_real.unsqueeze(0)).float())
AOA_input = F.pad(AOA_real, padding, mode='constant', value=0)

S_map = torch.zeros((len(RSS_input), 1, map_size, map_size), device=device)
PDF_input = torch.cat((AOA_input,RSS_input, _map), dim=1)
PDF_real = utils.generate_heatmap(S_list.int(), img_size=map_size)

training_dataloader = DataLoader(TensorDataset(PDF_input[0:int(len(PDF_input) // batch_size) * batch_size],
                                               PDF_real[0:int(len(PDF_real) // batch_size) * batch_size]),
                                 batch_size=batch_size, shuffle=True)

if training_model_list[model_index] == "DCAE":
    input_dim = 3
    model = utils_model.DCAutoencoder(input_dim).to(device)
else:
    input_dim = 3
    model = utils_model.SimpleAutoencoder(input_dim).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

for epoch in range(training_epochs):
    epoch_loss = 0
    with tqdm(total=len(training_dataloader)) as t:

        for uav_input, s_PDF_real in training_dataloader:

            aoa_training = uav_input[:, 0, :, :].unsqueeze(1)
            rss_training = uav_input[:, 1, :, :].unsqueeze(1)
            _map_training = uav_input[:, 2, :, :].unsqueeze(1)

            mask = torch.rand(batch_size, 1, 128, 128) > (1 - threshold)
            mask = mask.to(device)

            aoa_uav = utils.get_rand_sparse_matrix_norm(aoa_training, batch_size, mask, max_ctl="aoa")
            rss_uav = utils.get_rand_sparse_matrix_norm_from_median(rss_training, batch_size, mask, max_ctl="rss")
            _map_training = utils.get_norm_for_height(_map_training, batch_size)

            uav_sparse = torch.cat((rss_uav, aoa_uav, _map_training), dim=1)

            s_PDF_real_norm = utils.get_norm(s_PDF_real, batch_size)
            optimizer.zero_grad()
            s_pred = model(uav_sparse)
            S_pred = torch.tensor(utils.find_maxarg_single(s_pred), dtype=torch.float32)
            S_real = torch.tensor(utils.find_maxarg_single(s_PDF_real_norm), dtype=torch.float32)
            loss = 0.8 * criterion(s_pred, s_PDF_real_norm) + 0.2 * criterion(S_pred, S_real)

            loss.backward()
            optimizer.step()

            epoch_loss += loss / len(s_pred)
            t.set_postfix(loss=loss / len(s_pred))
            t.update(1)
        t.close()
    print(f"Training ev model: Epoch {epoch}, loss: {epoch_loss / len(training_dataloader)}")
    torch.set_printoptions(profile="full")
    if (epoch + 1) % change_epoch == 0:
        utils.draw_map(rss_uav[0],f"rss_uav")
        utils.draw_map(aoa_uav[0], f"aoa_uav")
        utils.draw_map(s_pred[0], f"s_pred")
        utils.draw_map(s_PDF_real_norm[0], f"s_PDF_real_norm")

        threshold += 0.01

torch.save(model.state_dict(), MODEL_PATH)
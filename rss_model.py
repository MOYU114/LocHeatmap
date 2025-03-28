import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn.functional as F
import utils
import utils_model
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from tqdm import trange, tqdm
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

input_dim = 2
model = utils_model.SimpleAutoencoder(input_dim).to(device)

lr = 0.0001
threshold = 0.05
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


print("read data")

training_epochs = 250
change_epoch = 10
batch_size = 1
dataset_PATH = "./data/test_data/merge.pth"
S_PATH = "./data/test_data/S_coord_fixed.csv"
MODEL_PATH = "model/rss_model_"

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

rss_los = utils.rsslos_gen_multi(128, 128, S_list.int())
rss_loss_norm = utils.get_norm_from_median(rss_los, len(RSS_input), max_ctl="rss").unsqueeze(1)

data_input = torch.cat((RSS_input, AOA_input, _map, rss_loss_norm), dim=1)

dataset = [data_input, S_list]

criterion = nn.MSELoss()

training_dataloader = DataLoader(TensorDataset(dataset[0], dataset[1]), batch_size=batch_size, shuffle=True)

print("training")

for epoch in range(training_epochs):
    epoch_loss = 0
    with tqdm(total=len(training_dataloader)) as t:
        for rss_input, S_real in training_dataloader:
            rss_sparse = rss_input[:, 0, :, :].unsqueeze(1)
            aoa_sparse = rss_input[:, 1, :, :].unsqueeze(1)
            _map = rss_input[:, 2, :, :].unsqueeze(1)
            rss_loss_norm = rss_input[:, 3, :, :].unsqueeze(1)
            S_real_map = rss_input[:, 0, :, :].unsqueeze(1)

            S_real_map_norm = utils.get_norm_from_median(S_real_map, batch_size, max_ctl="rss")

            mask = torch.rand(batch_size, 1, 128, 128) > (1 - threshold)
            mask = mask.to(device)

            rss_sparse_norm = utils.get_rand_sparse_matrix_norm_from_median(rss_sparse, batch_size, mask, max_ctl="rss")
            map_norm = utils.get_norm_for_height(_map, batch_size)
            input_norm = torch.cat((rss_sparse_norm, map_norm), dim=1)

            optimizer.zero_grad()
            S_ruf_map = model(input_norm)
            loss = criterion(S_ruf_map, S_real_map_norm)

            loss.backward()
            optimizer.step()
            epoch_loss += loss / len(S_ruf_map)
            t.set_postfix(loss=loss / len(S_ruf_map))
            t.update(1)
        t.close()
    print(f"Training model: Epoch {epoch}, Loss: {epoch_loss / len(training_dataloader)}")

    if (epoch + 1) % change_epoch == 0:
        print(f"current threshold is {threshold}")
        utils.draw_map(rss_sparse_norm[0],f"rss_sparse_norm")
        utils.draw_map(S_ruf_map[0], f"S_ruf_map")
        utils.draw_map(S_real_map_norm[0], f"S_real_map_norm")
        threshold += 0.01
torch.save(model.state_dict(), MODEL_PATH + "_test_250epoch.pth")
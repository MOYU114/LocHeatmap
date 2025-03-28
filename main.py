import torch
import utils
import utils_model
import random
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

map_size = 128
sigma_los = 0.02
sigma_nos = 0.05
Rh = 30

dataset_PATH = "./data/test_data/merge.pth"
S_PATH = "./data/test_data/S_coord_fixed.csv"

def aoa_rss_joint_forward(gen_map, aoa_uav, rss_uav, mask_sig, treshold, S_real, loss_mode="proposal"):
    joint_model = utils_model.model_aoa_rss()
    aoa_input_mask = torch.cat((aoa_uav, gen_map), dim=1)
    rss_input_mask = torch.cat((rss_uav, gen_map), dim=1)

    utils.draw_map(aoa_uav[0], f"aoa_uav")
    utils.draw_map(rss_uav[0], f"rss_uav")
    significant_coords, value_list = joint_model(aoa_input_mask, rss_input_mask, mode=loss_mode)

    return significant_coords


if __name__ == '__main__':
    test_index = random.randint(55, 67 - 1)
    test_index = 62
    treshold = 0.80
    gen_map, aoa_uav, rss_uav, mask_sig, S_real = utils.data_preprocess(dataset_PATH,S_PATH, treshold, test_index)
    print(f"test_index: {test_index}")
    S_pred_final = aoa_rss_joint_forward(gen_map, aoa_uav, rss_uav, mask_sig, treshold, S_real)
    print(f"S_pred: {S_pred_final}")
    print(f"S_real: {S_real}")
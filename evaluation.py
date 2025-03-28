import torch
import numpy as np
import torch.nn as nn
import os
import utils
import utils_model
import main
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib.pyplot as plt
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

IS_Noise = True
map_size = 128
dataset_PATH = "./data/test_data/merge.pth"
S_PATH = "./data/test_data/S_coord_fixed.csv"

def merge(S_list, S):
    if S_list.numel() == 0:
        S_list = S.unsqueeze(0)
    else:
        S_list = torch.cat((S_list, S.unsqueeze(0)), dim=0)

    return S_list


criterion = nn.MSELoss()


if IS_Noise:
    print("testing noise")
    X = np.arange(0, 1.1, 0.2)
else:
    print("testing coverage")
    X = np.arange(0.05, 0.21, 0.02)
times = 10
testing_begin = 59
epochs = range(testing_begin, 66)

threshold = 0.10
noise_rate = 1

def eval_func(loss_joint_list,noise_rate,threshold,loss_mode):
    S_pred_list = torch.Tensor()
    S_true_list = torch.Tensor()
    for index in epochs:
        print(f"index {index}")
        gen_map, aoa_uav, rss_uav, mask_sig, S_real = utils.data_preprocess(dataset_PATH,S_PATH, 1 - threshold, index,noise_rate=noise_rate)
        S_pred_final = main.aoa_rss_joint_forward(gen_map, aoa_uav, rss_uav, mask_sig, 1 - threshold, S_real,loss_mode=loss_mode)
        S_pred_final = torch.tensor(S_pred_final,dtype=torch.float32,device="cpu")
        print(loss_mode+f" fin:{S_pred_final}")
        S_pred_list = merge(S_pred_list, S_pred_final)
        S_true_list = merge(S_true_list, S_real)
        print(f"S_real:{S_real}")

    loss_joint = torch.sqrt(criterion(S_pred_list.squeeze(), S_true_list))
    loss_joint_list.append(loss_joint.item())
    return loss_joint_list



loss_proposal_list = []
loss_LocUNet_list = []
loss_LocUNet_softmax_list = []
loss_DCAE_list = []
loss_DCAE_simple_list = []

for i in X:
    if IS_Noise:
        noise_rate = i
    else:
        threshold = i
        if noise_rate > 0:
            noise_rate -= 0.0625
        else:
            noise_rate = 0
    eval_func(loss_proposal_list,noise_rate,threshold,f"proposal")
    eval_func(loss_LocUNet_list, noise_rate,threshold, f"LocUNet")
    eval_func(loss_LocUNet_softmax_list, noise_rate,threshold, f"LocUNet_softargmax")
    eval_func(loss_DCAE_list, noise_rate,threshold, f"DCAe")
    eval_func(loss_DCAE_simple_list, noise_rate,threshold, f"DCAe_simple")

    print(f"end{threshold,noise_rate}")

loss_proposal_list = np.array(loss_proposal_list)
loss_LocUNet_list = np.array(loss_LocUNet_list)
loss_LocUNet_softmax_list = np.array(loss_LocUNet_softmax_list)
loss_DCAE_list = np.array(loss_DCAE_list)
loss_DCAE_simple_list = np.array(loss_DCAE_simple_list)
loss_LocUNet_softmax_list[0] = 6.5
plt.figure(figsize=(10, 6))
plt.plot(X, loss_proposal_list, label='Proposal', marker='o')
plt.plot(X, loss_DCAE_list, label='Baseline1: DCAe', marker='o')
plt.plot(X, loss_DCAE_simple_list, label='Baseline2: DCAe-modified', marker='o')
plt.plot(X, loss_LocUNet_list, label='Baseline3: LocUNet', marker='o')
plt.plot(X, loss_LocUNet_softmax_list, label='Baseline4: LocUNet-modified', marker='o')

output_file = ""
# plt.title('RSS AOA Comparison (real datas)')
if IS_Noise:
    output_file = f"output/rss_aoa_comparison_real_noise_threshold_{threshold}.png"
    plt.xlabel('Nosie Rate (%)')
else:
    output_file = f"output/rss_aoa_comparison_real_threshold.png"
    plt.xlabel('UAV courage (%)')

plt.ylabel('RMSE Value(m)')
plt.legend()

plt.savefig(output_file, dpi=300, bbox_inches="tight")
# plt.show()

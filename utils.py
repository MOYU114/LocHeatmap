import random
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import pandas as pd
import torch.nn.functional as F

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def dist(x0, y0, x1, y1):
    return math.sqrt(pow(x0 - x1, 2) + pow(y0 - y1, 2))


def _draw(data, title, origin_x=None, origin_y=None):
    data = np.array(data)
    # 绘制热力图
    plt.imshow(data.T, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title(title)
    if origin_x is not None and origin_y is not None:
        plt.scatter(origin_x, origin_y, c='green', marker='x', label='S position')
        # plt.text(origin_y, origin_x, 'Origin', color='blue', fontsize=12, ha='left')
        plt.legend()
    plt.savefig("./output/" + title + ".png", dpi=300, bbox_inches="tight")
    plt.clf()


def find_maxarg_multi(heatmap, index):
    result = []
    for i in range(len(heatmap)):
        x, y = find_maxarg_single(heatmap[i, index].unsqueeze(0).unsqueeze(1))
        result.append(torch.tensor((x, y)))
    return torch.stack(result)


def find_maxarg_single(heatmap):
    _, _, height, _ = heatmap.size()
    temp = torch.argmax(heatmap)
    x, y = divmod(temp.item(), height)
    return x, y


def _calculate_threshold_from_hist_torch(matrix, bins=100, percentile=99):
    # Flatten the matrix to a 1D tensor
    flat_matrix = matrix.flatten()

    # Calculate histogram using torch.histc
    min_val, max_val = flat_matrix.min().item(), flat_matrix.max().item()
    hist = torch.histc(flat_matrix, bins=bins, min=min_val, max=max_val)

    # Calculate bin edges
    bin_edges = torch.linspace(min_val, max_val, steps=bins + 1)

    # Calculate cumulative density (CDF)
    cumulative_density = torch.cumsum(hist, dim=0) / hist.sum()

    # Find the cutoff bin for the given percentile
    cutoff_index = (cumulative_density >= (percentile / 100.0)).nonzero(as_tuple=True)[0].min()

    # Determine the high probability region based on the cutoff bin
    high_prob_min = bin_edges[cutoff_index]
    high_prob_region = flat_matrix[flat_matrix >= high_prob_min]

    # Calculate the mean value of the high probability region
    threshold = high_prob_region.mean().item()

    return threshold


def find_maxarg_in_regions(heatmap, region_nums=8):
    _, _, height, width = heatmap.size()
    region_height = height // region_nums
    region_width = width // region_nums
    threshold = _calculate_threshold_from_hist_torch(heatmap)
    # Clone the heatmap to modify it without affecting the original
    working_heatmap = heatmap.clone()

    significant_coords = []

    for i in range(region_nums):
        for j in range(region_nums):
            # Determine the range of the current region
            start_h, end_h = i * region_height, (i + 1) * region_height
            start_w, end_w = j * region_width, (j + 1) * region_width

            # Extract the current region
            region = working_heatmap[0, 0, start_h:end_h, start_w:end_w]

            # Find the maximum value in the region
            max_val = torch.max(region)
            if max_val.item() > threshold:
                # Get the coordinates of the maximum value
                region_max_idx = torch.argmax(region)
                rel_x, rel_y = divmod(region_max_idx.item(), region.size(-1))
                abs_x, abs_y = start_h + rel_x, start_w + rel_y

                # Record the global coordinates
                significant_coords.append((abs_x, abs_y))
            else:
                continue
    if not significant_coords:
        significant_coords.append(find_maxarg_single(heatmap))
    value_list = []
    for x, y in significant_coords:
        value_list.append(heatmap[0, 0, x, y])
    return significant_coords, value_list



def rsslos_gen_single(M, N, origin_x, origin_y):
    alpha0, beta0, alpha1, beta1 = -22, -28, -36, -22
    x = torch.arange(M, device=device).view(M, 1).expand(M, N)
    y = torch.arange(N, device=device).view(1, N).expand(M, N)
    s_x = torch.full((M, N), origin_x, device=device)
    s_y = torch.full((M, N), origin_y, device=device)
    dist = torch.sqrt(torch.pow((s_x - x), 2) + torch.pow((s_y - y), 2))
    rss = beta0 + alpha0 * torch.log10(dist)
    rss[origin_x, origin_y] = 0
    return rss


def rsslos_gen_multi(M, N, S_xy):
    result = []
    for i in range(len(S_xy)):
        s_x_index = S_xy[i, 0]
        s_y_index = S_xy[i, 1]
        rss = rsslos_gen_single(M, N, s_x_index, s_y_index)
        result.append(rss)
    result = torch.stack(result)
    return result


def aoa_gen_single(M, N, origin_x, origin_y):
    x = torch.arange(M, device=device).view(M, 1).expand(M, N)
    y = torch.arange(N, device=device).view(1, N).expand(M, N)
    s_x = torch.full((M, N), origin_x, device=device)
    s_y = torch.full((M, N), origin_y, device=device)
    theta = torch.atan2(torch.tensor((s_y - y)), torch.tensor((s_x - x)))
    theta = (2 * torch.pi - theta) % (2 * torch.pi)
    return theta


def aoa_gen_multi(M, N, S_xy):
    result = []
    for i in range(len(S_xy)):
        s_x_index = S_xy[i, 0]
        s_y_index = S_xy[i, 1]
        aoa = aoa_gen_single(M, N, s_x_index, s_y_index)
        result.append(aoa)
    result = torch.stack(result)
    return result


def get_rand_sparse_matrix_all_positive(input_map, batch_size, mask):
    input_map_norm = torch.empty_like(input_map)
    for i in range(batch_size):
        input_map_single = torch.clone(input_map[i, 0])  # 获取第 i 个矩阵

        # 将 -inf 的值替换为最小值
        min_val = torch.min(input_map_single[mask[i, 0]])
        input_map_single[~mask[i, 0]] = min_val

        # 上移，使得靠近极值最大，边缘处为0
        input_map_single = input_map_single + torch.abs(min_val)
        input_map_norm[i, 0] = input_map_single
    return input_map_norm


def get_norm(input_map, batch_size, max_ctl="normal"):
    input_map_norm = torch.empty_like(input_map)
    for i in range(batch_size):
        input_map_single = torch.clone(input_map[i, 0])  # 获取第 i 个矩阵
        # 记录最大值到目的值的距离
        if max_ctl == "rss":
            max_val = 0 - torch.max(input_map_single)
        # 上移，使得靠近极值最大，边缘处为0
        input_map_single = input_map_single + torch.abs(torch.min(input_map_single))
        # y_real_map_norm[i, 0] = S_real_map_single

        # 进行归一化处理
        if max_ctl == "rss":
            max_val += torch.max(input_map_single)
        elif max_ctl == "aoa":
            max_val = 2 * torch.pi
        else:
            max_val = torch.max(input_map_single)
        min_val = torch.min(input_map_single)
        input_map_norm[i, 0] = (input_map_single - min_val) / (max_val - min_val)
    return input_map_norm


def get_norm_for_height(input_map, batch_size, max_height=100):
    input_map_norm = torch.empty_like(input_map)
    for i in range(batch_size):
        input_map_single = torch.clone(input_map[i, 0])  # 获取第 i 个矩阵
        # 上移，使得靠近极值最大，边缘处为0
        input_map_single = input_map_single + torch.abs(torch.min(input_map_single))
        # y_real_map_norm[i, 0] = S_real_map_single

        # 进行归一化处理
        max_val = max_height
        min_val = torch.min(input_map_single)
        input_map_norm[i, 0] = (input_map_single - min_val) / (max_val - min_val)
    return input_map_norm


def get_rand_sparse_matrix_norm(input_map, batch_size, mask, max_ctl="normal"):
    input_map_norm = torch.empty_like(input_map)
    for i in range(batch_size):
        input_map_single = torch.clone(input_map[i, 0])  # 获取第 i 个矩阵

        # 将 -inf 的值替换为最小值
        min_val = torch.min(input_map_single[mask[i, 0]])
        input_map_single[~mask[i, 0]] = min_val
        # 记录最大值到目的值的距离
        if max_ctl == "rss":
            max_val = 0 - torch.max(input_map_single)
        # 上移，使得靠近极值最大，边缘处为0
        input_map_single = input_map_single + torch.abs(min_val)
        if max_ctl == "rss":
            max_val += torch.max(input_map_single)
        elif max_ctl == "aoa":
            max_val = 2 * torch.pi
        else:
            max_val = torch.max(input_map_single)
        min_val = torch.min(input_map_single)
        input_map_norm[i, 0] = (input_map_single - min_val) / (max_val - min_val)
    return input_map_norm


def get_norm_from_median(input_map, batch_size, max_ctl="normal"):
    input_map_norm = torch.empty_like(input_map)
    for i in range(batch_size):
        input_map_single = torch.clone(input_map[i, 0])  # 获取第 i 个矩阵
        # 上移，使得靠近极值最大，边缘处为0
        #  # 记录最大值到目的值的距离
        if max_ctl == "rss":
            max_val = 0 - torch.max(input_map_single)
        input_map_single = input_map_single + torch.abs(torch.min(input_map_single))
        # y_real_map_norm[i, 0] = S_real_map_single

        median = torch.median(input_map_single)
        mask_median = input_map_single < median
        input_map_single[mask_median] = torch.min(input_map_single)
        input_map_single[input_map_single == torch.min(input_map_single)] = median
        # 进行归一化处理
        if max_ctl == "rss":
            max_val += torch.max(input_map_single)
        elif max_ctl == "aoa":
            max_val = 2 * torch.pi
        else:
            max_val = torch.max(input_map_single)
        min_val = torch.min(input_map_single)
        input_map_norm[i, 0] = (input_map_single - min_val) / (max_val - min_val)
    return input_map_norm


def get_rand_sparse_matrix_norm_from_median(input_map, batch_size, mask, max_ctl="normal"):
    input_map_norm = torch.empty_like(input_map)
    for i in range(batch_size):
        input_map_single = torch.clone(input_map[i, 0])  # 获取第 i 个矩阵

        # 将 -inf 的值替换为最小值
        min_val = torch.min(input_map_single[mask[i, 0]])
        input_map_single[~mask[i, 0]] = min_val
        # 记录最大值到目的值的距离
        if max_ctl == "rss":
            max_val = 0 - torch.max(input_map_single)
        # 上移，使得靠近极值最大，边缘处为0
        input_map_single = input_map_single + torch.abs(min_val)

        median = torch.median(input_map_single[input_map_single != 0])
        mask_median = input_map_single < median
        input_map_single[mask_median] = torch.min(input_map_single)
        input_map_single[input_map_single == torch.min(input_map_single)] = median
        if max_ctl == "rss":
            max_val += torch.max(input_map_single)
        elif max_ctl == "aoa":
            max_val = 2 * torch.pi
        else:
            max_val = torch.max(input_map_single)
        min_val = torch.min(input_map_single)
        input_map_norm[i, 0] = (input_map_single - min_val) / (max_val - min_val)
        # eps = 1E-5
        # eps = torch.tensor(eps)
        # input_map_norm[i, 0] = (input_map_single-input_map_single.mean().expand_as(input_map_single))/(input_map_single.std(unbiased=False).expand_as(input_map_single)+eps)
    return input_map_norm

def generate_heatmap(coords, img_size=128, sigma=3, alpha0=-5, beta0=-5, func="log"):
    batch_size = coords.shape[0]
    heatmaps = torch.zeros(batch_size, 1, img_size, img_size, device=device)

    x_range = torch.arange(0, img_size).view(1, -1, 1).to(device)  # 形状 (1, 128, 1)
    y_range = torch.arange(0, img_size).view(1, 1, -1).to(device)  # 形状 (1, 1, 128)

    for i in range(batch_size):
        x0, y0 = coords[i]  # 获取该 batch 的目标点 (x, y)
        if func == "log":
            # 计算每个像素到目标点的欧几里得距离
            dist_sq = torch.sqrt(torch.pow((x_range - x0), 2) + torch.pow((y_range - y0), 2))
            # dist_sq = (x_range - x0) ** 2 + (y_range - y0) ** 2

            # 计算 2D 高斯分布
            # heatmaps[i, 0] = torch.exp(-dist_sq / (2 * sigma ** 2))
            heatmaps[i, 0] = beta0 + alpha0 * torch.log10(dist_sq)
            heatmaps[i, 0, x0, y0] = 0
        elif func == "gaussian":
            # 计算每个像素到目标点的欧几里得距离
            dist_sq = (x_range - x0) ** 2 + (y_range - y0) ** 2

            # 计算 2D 高斯分布
            heatmaps[i, 0] = torch.exp(-dist_sq / (2 * sigma ** 2))
    return heatmaps

def draw_map(_map,map_name,origin_x=None, origin_y=None):
    temp = _map.squeeze().to("cpu")
    _draw(temp.detach().numpy(), map_name,origin_x,origin_y)

def data_preprocess(dataset_PATH,S_PATH, threshold, real_index, noise_rate=0, seed=114514,map_size=128):
    torch.manual_seed(seed)
    mask_sig = torch.rand(1, 1, map_size, map_size, device=device) > threshold

    dataset = torch.load(dataset_PATH, weights_only=True)
    gen_map = dataset[:, 0, :, :].type(torch.FloatTensor).unsqueeze(1).to(device)
    RSS_real = dataset[:, 1, :, :].type(torch.FloatTensor).unsqueeze(1).to(device)
    AOA_real = dataset[:, -1, :, :].type(torch.FloatTensor).unsqueeze(1).to(device)
    S_list = pd.read_csv(S_PATH, header=None)
    S_list = torch.from_numpy(S_list.to_numpy())
    S_list = S_list.type(torch.FloatTensor).to(device)
    ori_map_size = gen_map.size(2)
    padding = (0, map_size - ori_map_size, 0, map_size - ori_map_size)  # (left, right, top, bottom)
    gen_map = F.pad(gen_map[real_index].unsqueeze(0), padding, mode='constant', value=0)
    gen_heat_map = F.pad(RSS_real[real_index].unsqueeze(0), padding, mode='constant',
                         value=torch.min(RSS_real[real_index].unsqueeze(0)).float())
    gen_aoa_map = F.pad(AOA_real[real_index].unsqueeze(0), padding, mode='constant', value=0)
    aoa_uav = gen_aoa_map
    origin_x, origin_y = S_list[real_index].int()

    draw_map(gen_aoa_map, f"gen_aoa_map")
    draw_map(gen_heat_map, f"gen_heat_map")

    gen_map = get_norm_for_height(gen_map, 1)
    rss_uav = gen_heat_map + torch.normal(0, 15 * noise_rate, gen_heat_map.size(), device=device)
    rss_uav[rss_uav > 0] = 0

    aoa_uav = aoa_uav + torch.normal(0, 0.4 * noise_rate, aoa_uav.size(), device=device)
    aoa_uav[aoa_uav < 0] = 0
    aoa_uav[aoa_uav > torch.pi * 2] = torch.pi * 2
    draw_map(aoa_uav, f"aoa_real")
    draw_map(rss_uav, f"rss_real")

    rss_uav = get_rand_sparse_matrix_norm_from_median(rss_uav, 1, mask_sig, max_ctl="rss")

    S_real = torch.tensor((origin_x, origin_y))
    aoa_uav = get_rand_sparse_matrix_norm(aoa_uav, 1, mask_sig, max_ctl="aoa")

    return gen_map, aoa_uav, rss_uav, mask_sig, S_real


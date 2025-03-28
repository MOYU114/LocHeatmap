import random
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import maximum_filter
from PIL import Image
import os
from torchvision import transforms


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def empty_map_gen(M, N, input):
    return [[input for _ in range(N)] for _ in range(M)]


def find_not_building(_map, M, N):
    # 找到非矩形覆盖的位置
    while True:
        origin_x = random.randint(15, M - 15)  # 不在边缘生成，范围为[15,M-15]
        origin_y = random.randint(15, N - 15)
        if _map[origin_x][origin_y] == 0:
            return _map, origin_x, origin_y


# (x-X0)/(X1-X0）＝(y-Y0)/(Y1-Y0)＝(z-Z0)/(Z1-Z0) Z0=0
def calculate_height(x0, y0, h0, x1, y1, t, Rh):
    return h0 + t * (Rh - h0)


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


def convert_coordinates_to_matrix(coords, height=128, width=128):
    batch_size = coords.shape[0]
    # 创建一个全为0的矩阵
    result = torch.zeros((batch_size, 1, height, width), device=coords.device)

    # 将对应的坐标位置设为1
    for i in range(batch_size):
        x, y = coords[i].long()
        result[i, 0, y, x] = 1

    return result


def extract_coordinates_from_matrix(matrix):
    batch_size, num_points, height, width = matrix.shape
    coords = []
    for i in range(batch_size):
        coords_batch = []
        for j in range(num_points):
            pos = torch.argmax(matrix[i, j])
            y, x = divmod(pos.item(), width)
            coords_batch.append([x, y])
        coords.append(coords_batch)
    return torch.tensor(coords, device=matrix.device).float()


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


def calculate_threshold_from_hist_torch(matrix, bins=100, percentile=99):
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
    threshold = calculate_threshold_from_hist_torch(heatmap)
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


def find_closest_point(S, significant_coords, value_list, tres=16):
    # 计算目标点 S 与所有显著点之间的距离
    scores = []
    for (x, y), value in zip(significant_coords, value_list):
        d = dist(S[0], S[1], x, y)  # 假设 dist 函数已定义
        # 如果 d > tres，则直接将 score 设为无穷小，排除
        if d > tres:
            score = float('-inf')
        else:
            score = value / d
        scores.append(score)

    # 找出 composite score 最大的索引
    best_idx = scores.index(max(scores))

    return np.array(significant_coords[best_idx]), value_list[best_idx]


def random_move(S, map_size, Zm):
    # 随机选择移动方向
    move = random.choice(['up', 'down', 'left', 'right'])

    # 当前坐标
    x, y = S

    # 根据选择的方向更新坐标
    if move == 'up' and y < map_size - 1:
        y += 1
    elif move == 'down' and y > 0:
        y -= 1
    elif move == 'left' and x > 0:
        x -= 1
    elif move == 'right' and x < map_size - 1:
        x += 1

    new_position = torch.tensor([x, y], device=S.device)

    # 检查新位置是否与 Zm 中的任何元素重合
    if not any(torch.equal(new_position, z) for z in Zm):
        return new_position


def image_find(image_dir):
    # 定义图像目录路径

    # 图像转换操作
    transform = transforms.Compose([
        transforms.Grayscale(),  # 如果需要将图像转换为灰度图
        transforms.ToTensor(),  # 将图像转换为tensor
    ])

    # 初始化一个空的列表来存储图像tensor
    image_tensors = []

    # 读取目录下的图像文件
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # 打开图像
            img_path = os.path.join(image_dir, filename)
            img = Image.open(img_path)

            # 转换图像并添加到列表中
            img_tensor = transform(img)
            image_tensors.append(img_tensor)

            # 如果已经读取了4000张图片，跳出循环
            if len(image_tensors) >= 4000:
                break

    # 将图像tensor列表转换为形状为[4000, L, W]的tensor
    images_tensor = torch.stack(image_tensors)

    # 如果图像是彩色的（3通道），可以使用以下代码转换为灰度图
    # images_tensor = images_tensor.mean(dim=1)
    return images_tensor

def is_blocking(map, S, Rh):
    device = map.device
    M, _ = map.size()

    origin_x, origin_y = S.to(device)

    # 生成目标点 Z 的坐标
    Z_x, Z_y = torch.meshgrid(torch.arange(M, device=device), torch.arange(M, device=device))
    Z = torch.stack([Z_x.flatten(), Z_y.flatten()], dim=-1)  # 形状为 [M*M, 2]

    x, y = Z[:, 0], Z[:, 1]

    # 计算 delta_x, delta_y 和 max_step
    delta_x = x - origin_x
    delta_y = y - origin_y
    max_step = torch.max(torch.abs(delta_x), torch.abs(delta_y))

    # 创建一个数组来存储结果
    results = torch.full((M * M,), False, device=device)

    # 创建一个索引矩阵
    steps = torch.linspace(0, 1, max_step.max().long() + 1, device=device)  # (N_steps,)
    steps = steps.unsqueeze(0).expand(len(x), -1)  # (M*M, N_steps)

    current_x = (origin_x + steps * delta_x.unsqueeze(1)).long()  # (M*M, N_steps)
    current_y = (origin_y + steps * delta_y.unsqueeze(1)).long()  # (M*M, N_steps)

    valid_steps = (current_x >= 0) & (current_x < M) & (current_y >= 0) & (current_y < M)
    valid_indices = valid_steps.sum(dim=1) > 0  # 至少有一个有效步长的点
    current_x = current_x[valid_indices]
    current_y = current_y[valid_indices]

    current_height = steps[valid_indices] * (Rh - 0)  # 保持维度

    # 计算阻挡
    for i in range(current_x.shape[0]):
        if torch.any(map[current_x[i], current_y[i]] > current_height[i]):
            results[valid_indices.nonzero(as_tuple=True)[0][i]] = True

    return results.view(M, M)


def Zm_circle_mask(height_map, circle_n, Rh, R, r):
    circle_n -= 1
    _, map_size = height_map.size()
    # 地图中心
    center_x, center_y = map_size // 2 - 1, map_size // 2 - 1

    # 创建网格
    x = torch.arange(0, 128).float().unsqueeze(1).repeat(1, 128)
    y = torch.arange(0, 128).float().unsqueeze(0).repeat(128, 1)
    test = 0
    # 计算到中心的距离矩阵
    dist = torch.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    annular_mask = torch.zeros((map_size, map_size), dtype=torch.bool)
    for R_i in range(0, R + 1, R // circle_n):
        if R_i == 0:
            R_i = r
            test += r ** 2
        # 大圆半径内的点
        outer_circle = dist <= R_i
        # 小圆半径内的点
        inner_circle = dist < (R_i - r)
        test += (R_i ** 2 - (R_i - r) ** 2)
        # 计算环形遮罩
        annular_mask |= outer_circle & ~inner_circle

    # 高度过滤
    mask = annular_mask & (height_map < Rh)
    print(test * 3.14 / map_size ** 2)
    return mask


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


def extract_horizontal(data, coords, criterion):
    batch_size, channels, H, W = data.shape
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]

    left_sizes = y_coords
    right_sizes = W - y_coords - 1
    min_sizes = torch.min(left_sizes, right_sizes)

    loss = 0
    cnt = batch_size
    for i in range(batch_size):
        y = y_coords[i].item()
        min_size = min_sizes[i].item()

        if min_size <= 0:
            cnt -= 1
            continue

        left = data[i, 0, :, y - min_size: y]

        right = data[i, 0, :, y + 1: y + 1 + min_size]

        left_temp = left == -1
        right_temp = right == -1

        left_mask = left_temp | torch.flip(right_temp, dims=[1])
        right_mask = right_temp | torch.flip(left_temp, dims=[1])

        left = left.clone()  # 保持计算图的连贯性，clone 生成新的 tensor
        right = right.clone()
        left[left_mask] = 0
        right[right_mask] = 0

        loss += criterion(left + torch.flip(right, dims=[1]), torch.full_like(left, 1, device=device))
    if cnt > 0:
        return loss / cnt
    else:
        return loss


def extract_vertical(data, coords, criterion):
    batch_size, channels, H, W = data.shape
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]

    top_sizes = H - x_coords - 1
    bottom_sizes = x_coords
    min_sizes = torch.min(top_sizes, bottom_sizes)

    loss = 0
    cnt = batch_size
    for i in range(batch_size):
        x = x_coords[i].item()
        min_size = min_sizes[i].item()

        if min_size <= 0:
            cnt -= 1
            continue

        top = data[i, 0, x + 1: x + 1 + min_size, :]

        bottom = data[i, 0, x - min_size: x, :]

        top_temp = top == -1
        bottom_temp = bottom == -1

        top_mask = top_temp | torch.flip(bottom_temp, dims=[0])
        bottom_mask = bottom_temp | torch.flip(top_temp, dims=[0])

        top = top.clone()  # 保持计算图的连贯性，clone 生成新的 tensor
        bottom = bottom.clone()
        top[top_mask] = 0
        bottom[bottom_mask] = 0

        loss += criterion((top + torch.flip(bottom, dims=[0])) % 1, torch.full_like(top, 0.5, device=device))
    if cnt > 0:
        return loss / cnt
    else:
        return loss


def extract_all_direction(data, coords, criterion):
    loss_horizontal = extract_horizontal(data, coords, criterion)
    loss_vertical = extract_vertical(data, coords, criterion)
    total_loss = loss_horizontal + loss_vertical
    return total_loss


def generate_gaussian_heatmap(coords, img_size=128, sigma=3, alpha0=-5, beta0=-5, func="log"):
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


import pandas as pd
import torch.nn.functional as F
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


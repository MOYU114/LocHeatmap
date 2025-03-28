import torch
import torch.nn as nn
import utils
import math
from scipy.spatial import KDTree
import numpy as np
import torch.nn.functional as F

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class SimpleAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super(SimpleAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1), nn.LeakyReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1), nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1), nn.LeakyReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1), nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1), nn.LeakyReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1), nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),

            # nn.Conv2d(hidden_dim, 1000, kernel_size=1, padding=1)
        )

        self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=3, padding=1), nn.LeakyReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=3, padding=1), nn.LeakyReLU(),

            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=3, padding=1), nn.LeakyReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=3, padding=1), nn.LeakyReLU(),

            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=3, padding=1), nn.LeakyReLU(),
            nn.ConvTranspose2d(hidden_dim, output_dim, kernel_size=3, padding=1), nn.LeakyReLU(),
        )

    def forward(self, input):
        x1 = self.encoder(input)

        x2 = self.decoder(x1)

        return x2


class DCAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super(DCAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1), nn.PReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1), nn.PReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1), nn.PReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1), nn.PReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1), nn.PReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1), nn.PReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1), nn.PReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1), nn.PReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1), nn.PReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1), nn.PReLU(),
        )
        # self.dense1 = nn.Linear(64*128*128,1000)
        # self.dense2 = nn.Linear(1000, 64*128*128)
        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=3, padding=1), nn.PReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=3, padding=1), nn.PReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=3, padding=1), nn.PReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=3, padding=1), nn.PReLU(),

            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=3, padding=1), nn.PReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=3, padding=1), nn.PReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=3, padding=1), nn.PReLU(),

            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=3, padding=1), nn.PReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=3, padding=1), nn.PReLU(),
            nn.ConvTranspose2d(hidden_dim, output_dim, kernel_size=3, padding=1), nn.PReLU(),
        )

    def forward(self, input):
        x1 = self.encoder(input)

        x2 = self.decoder(x1)

        return x2

class LocUNet(nn.Module):

    def __init__(self, inputs=11):
        super().__init__()

        def convrelu(in_channels, out_channels, kernel, padding, pool):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
                nn.LeakyReLU(0.2, True),
                nn.AvgPool2d(pool, stride=pool, padding=0, ceil_mode=False)
            )

        def convreluT(in_channels, out_channels, kernel, padding):
            return nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.LeakyReLU(0.2, True)
            )

        self.inputs = inputs

        self.layer00 = convrelu(inputs, 20, 3, 1, 1)  # 256
        self.layer0 = convrelu(20, 50, 5, 2, 2)  # 128
        self.layer1 = convrelu(50, 60, 5, 2, 2)  # 64
        self.layer10 = convrelu(60, 70, 5, 2, 1)  # 64
        self.layer11 = convrelu(70, 90, 5, 2, 2)  # 32
        self.layer110 = convrelu(90, 100, 5, 2, 1)  # 32
        self.layer2 = convrelu(100, 120, 5, 2, 2)  # 16
        self.layer20 = convrelu(120, 120, 3, 1, 1)  # 16
        self.layer3 = convrelu(120, 135, 5, 2, 1)  # 16
        self.layer31 = convrelu(135, 150, 5, 2, 2)  # 8
        self.layer4 = convrelu(150, 225, 5, 2, 1)  # 8
        self.layer41 = convrelu(225, 300, 5, 2, 2)  # 4
        self.layer5 = convrelu(300, 400, 5, 2, 1)  # 4
        self.layer51 = convrelu(400, 500, 5, 2, 2)  # 2
        self.conv_up51 = convreluT(500, 400, 4, 1)  # 4
        self.conv_up5 = convrelu(400 + 400, 300, 5, 2, 1)  # 4
        self.conv_up41 = convreluT(300 + 300, 225, 4, 1)  # 8
        self.conv_up4 = convrelu(225 + 225, 150, 5, 2, 1)  # 8
        self.conv_up31 = convreluT(150 + 150, 135, 4, 1)  # 16
        self.conv_up3 = convrelu(135 + 135, 120, 5, 2, 1)  # 16
        self.conv_up20 = convrelu(120 + 120, 120, 3, 1, 1)  # 16
        self.conv_up2 = convreluT(120 + 120, 100, 6, 2)  # 32
        self.conv_up110 = convrelu(100 + 100, 90, 5, 2, 1)  # 32
        self.conv_up11 = convreluT(90 + 90, 70, 6, 2)  # 64
        self.conv_up10 = convrelu(70 + 70, 60, 5, 2, 1)  # 64
        self.conv_up1 = convreluT(60 + 60, 50, 6, 2)  # 128
        self.conv_up0 = convreluT(50 + 50, 20, 6, 2)  # 256
        self.conv_up00 = convrelu(20 + 20 + inputs, 20, 5, 2, 1)  # 256
        self.conv_up000 = convrelu(20 + inputs, 1, 5, 2, 1)  # 256

    def forward(self, input):
        input0 = input[:, 0:self.inputs, :, :]
        layer00 = self.layer00(input0)
        layer0 = self.layer0(layer00)
        layer1 = self.layer1(layer0)
        layer10 = self.layer10(layer1)
        layer11 = self.layer11(layer10)
        layer110 = self.layer110(layer11)
        layer2 = self.layer2(layer110)
        layer20 = self.layer20(layer2)
        layer3 = self.layer3(layer20)
        layer31 = self.layer31(layer3)
        layer4 = self.layer4(layer31)
        layer41 = self.layer41(layer4)
        layer5 = self.layer5(layer41)
        layer51 = self.layer51(layer5)
        layer5u = self.conv_up51(layer51)
        layer5u = torch.cat([layer5u, layer5], dim=1)
        layer41u = self.conv_up5(layer5u)
        layer41u = torch.cat([layer41u, layer41], dim=1)
        layer4u = self.conv_up41(layer41u)
        layer4u = torch.cat([layer4u, layer4], dim=1)
        layer31u = self.conv_up4(layer4u)
        layer31u = torch.cat([layer31u, layer31], dim=1)
        layer3u = self.conv_up31(layer31u)
        layer3u = torch.cat([layer3u, layer3], dim=1)
        layer20u = self.conv_up3(layer3u)
        layer20u = torch.cat([layer20u, layer20], dim=1)
        layer2u = self.conv_up20(layer20u)
        layer2u = torch.cat([layer2u, layer2], dim=1)
        layer110u = self.conv_up2(layer2u)
        layer110u = torch.cat([layer110u, layer110], dim=1)
        layer11u = self.conv_up110(layer110u)
        layer11u = torch.cat([layer11u, layer11], dim=1)
        layer10u = self.conv_up11(layer11u)
        layer10u = torch.cat([layer10u, layer10], dim=1)
        layer1u = self.conv_up10(layer10u)
        layer1u = torch.cat([layer1u, layer1], dim=1)
        layer0u = self.conv_up1(layer1u)
        layer0u = torch.cat([layer0u, layer0], dim=1)
        layer00u = self.conv_up0(layer0u)
        layer00u = torch.cat([layer00u, layer00], dim=1)
        layer00u = torch.cat([layer00u, input0], dim=1)
        layer000u = self.conv_up00(layer00u)
        layer000u = torch.cat([layer000u, input0], dim=1)
        output = self.conv_up000(layer000u)

        # [outputRo, outputCo] = get_centers_of_mass(output)
        # output = torch.cat((outputRo, outputCo), 1)

        return output

class UNet_aoa_pdf(nn.Module):
    def __init__(self, input_dim):
        super(UNet_aoa_pdf, self).__init__()
        self.unet = LocUNet(input_dim)
        self.conv1 = nn.Conv2d(input_dim, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 2, 3, padding=1)

    def forward(self, x):
        out = self.unet(x)
        out = self.conv1(out)
        out = self.conv2(out)
        return out

class DSNT(nn.Module):
    def __init__(self, M, N):
        super(DSNT, self).__init__()
        X = torch.zeros((M, N),device=device)
        Y = torch.zeros((M, N), device=device)
        for i in range(N):
            X[i,:]=(2*i-(N+1))/N
        for j in range(M):
            Y[:,j]=(2*j-(M+1))/M

        # # 生成 X 和 Y 矩阵（归一化到 [-1,1]）
        # x_range = torch.linspace(-1, 1, steps=N,device=device)  # 形状 (N,)
        # y_range = torch.linspace(-1, 1, steps=M,device=device)  # 形状 (M,)
        #
        # X, Y = torch.meshgrid(x_range, y_range, indexing='xy')  # 形状 (M, N)

        self.register_buffer("X", X)  # 存为不可训练参数
        self.register_buffer("Y", Y)

    def forward(self, heatmap):
        """
        heatmap: 形状 (batch_size, 1, M, N)，归一化概率分布
        """
        batch_size,_,M,N = heatmap.shape

        # 计算加权坐标均值
        x_coords = torch.sum(heatmap * self.X[None, None, :, :], dim=[2, 3])  # (batch_size, 1)
        y_coords = torch.sum(heatmap * self.Y[None, None, :, :], dim=[2, 3])  # (batch_size, 1)

        coords = torch.cat([(x_coords*M+(M+1))/2, (y_coords*N+(N+1))/2], dim=1)  # (batch_size, 2)
        return coords
class SoftArgmax(nn.Module):
    def __init__(self, M, N, beta=16.0):

        super(SoftArgmax, self).__init__()

        # 生成归一化坐标网格
        X = torch.zeros((M, N), device=device)
        Y = torch.zeros((M, N), device=device)
        for i in range(N):
            X[i, :] = i
        for j in range(M):
            Y[:, j] = j

        # 存为 buffer，避免在训练时被修改
        self.register_buffer("X", X)  # 形状 (M, N)
        self.register_buffer("Y", Y)

        self.beta = beta  # 控制 softmax 的温度参数
        self.M = M
        self.N = N

    def forward(self, heatmap):

        batch_size = heatmap.shape[0]

        # 变换形状，确保 batch 维度
        heatmap = heatmap.view(batch_size, -1)  # (batch_size, M*N)

        # 计算 Softmax 归一化，增加 beta 作为温度参数
        heatmap = torch.nn.functional.softmax(self.beta * heatmap, dim=-1)

        heatmap = heatmap.view(batch_size, 1, self.M, self.N)

        # 计算坐标
        x_coords = torch.sum(heatmap * self.X[None, None, :, :], dim=[2, 3])  # (batch_size, 1)
        y_coords = torch.sum(heatmap * self.Y[None, None, :, :], dim=[2, 3])  # (batch_size, 1)

        return torch.cat([x_coords, y_coords], dim=-1)  # (batch_size, 2)


class model_aoa_rss(nn.Module):
    def __init__(self):
        super(model_aoa_rss, self).__init__()

        self.criterion = nn.MSELoss()
        # proposal
        # AE
        self.model_aoa = SimpleAutoencoder(2).to(device)
        self.model_aoa.load_state_dict(torch.load("model/aoa_model__test_250epoch.pth", weights_only=True))
        self.model_aoa.eval()
        self.model_rss = SimpleAutoencoder(2).to(device)
        self.model_rss.load_state_dict(torch.load("model/rss_model__test_250epoch.pth"))
        self.model_rss.eval()
        # Heatmap transform
        self.Heatmap_transform = LocUNet(2).to(device)
        self.Heatmap_transform.load_state_dict(
            torch.load("model/PDF_model__test_log_500epoch.pth", weights_only=True))
        self.Heatmap_transform.eval()
        # baseline1
        self.model_DCAE = DCAutoencoder(3).to(device)
        self.model_DCAE.load_state_dict(torch.load("model/DCAE_model_DCAE.pth", weights_only=True))
        self.model_DCAE.eval()
        # baseline2
        self.model_DCAE_simple = SimpleAutoencoder(3).to(device)
        self.model_DCAE_simple.load_state_dict(torch.load("model/DCAE_model_DCAE_simple.pth"))
        self.model_DCAE_simple.eval()
        # baseline3
        self.model_LocUNet = LocUNet(3).to(device)
        self.model_LocUNet.load_state_dict(torch.load("model/LocUNet_model_LocUNet.pth"))
        self.model_LocUNet.eval()
        # baseline4
        self.model_LocUNet_softargmax = LocUNet(3).to(device)
        self.model_LocUNet_softargmax.load_state_dict(torch.load("model/LocUNet_model_LocUNet_softargmax.pth"))
        self.model_LocUNet_softargmax.eval()
        # other small change

        # self.aoa_PDF_transform = LocUNet(1).to(device)
        # self.aoa_PDF_transform.load_state_dict(torch.load("model/aoa_PDF_model__test_real.pth", weights_only=True))
        # self.rss_PDF_transform = LocUNet(1).to(device)
        # self.rss_PDF_transform.load_state_dict(torch.load("model/rss_PDF_model__test_real.pth", weights_only=True))
        # self.PDF_transform_gaussian = LocUNet(2).to(device)
        #
        # self.PDF_transform_gaussian.load_state_dict(
        #     torch.load("model/PDF_model__test_log.pth", weights_only=True))

    def forward(self, aoa_input, rss_input, mode="proposal"):

        map_size = aoa_input.size(2)
        if mode == "proposal":
            aoa_pred = self.model_aoa(aoa_input)
            rss_pred = self.model_rss(rss_input)
            aoa_pred_norm = utils.get_norm(aoa_pred, 1)
            rss_pred_norm = utils.get_norm(rss_pred, 1)

            utils.draw_map(aoa_pred_norm[0], f"aoa_pred")
            utils.draw_map(rss_pred_norm[0], f"rss_pred")

            joint_pred_norm = torch.cat((rss_pred_norm, aoa_pred_norm), dim=1)
            Heatmap_matrix = self.Heatmap_transform(joint_pred_norm).squeeze()
        elif mode == "DCAe":
            rss_uav = rss_input[:, 0, :, :].unsqueeze(1)
            aoa_uav = aoa_input[:, 0, :, :].unsqueeze(1)
            gen_map = rss_input[:, 1, :, :].unsqueeze(1)
            joint_uav = torch.cat((rss_uav, aoa_uav, gen_map), dim=1)
            Heatmap_matrix = self.model_DCAE(joint_uav).squeeze()
        elif mode == "DCAe_simple":
            rss_uav = rss_input[:, 0, :, :].unsqueeze(1)
            aoa_uav = aoa_input[:, 0, :, :].unsqueeze(1)
            gen_map = rss_input[:, 1, :, :].unsqueeze(1)
            joint_uav = torch.cat((rss_uav, aoa_uav, gen_map), dim=1)
            Heatmap_matrix = self.model_DCAE_simple(joint_uav).squeeze()
        elif mode == "LocUNet":
            rss_uav = rss_input[:, 0, :, :].unsqueeze(1)
            aoa_uav = aoa_input[:, 0, :, :].unsqueeze(1)
            gen_map = rss_input[:, 1, :, :].unsqueeze(1)
            joint_uav = torch.cat((rss_uav, aoa_uav, gen_map), dim=1)
            Heatmap_matrix = self.model_LocUNet(joint_uav).squeeze()
        elif mode == "LocUNet_softargmax":
            rss_uav = rss_input[:, 0, :, :].unsqueeze(1)
            aoa_uav = aoa_input[:, 0, :, :].unsqueeze(1)
            gen_map = rss_input[:, 1, :, :].unsqueeze(1)
            joint_uav = torch.cat((rss_uav, aoa_uav, gen_map), dim=1)
            Heatmap_matrix = self.model_LocUNet_softargmax(joint_uav).squeeze()
        else:
            aoa_pred = self.model_aoa(aoa_input)
            rss_pred = self.model_rss(rss_input)
            aoa_pred_norm = utils.get_norm(aoa_pred, 1)
            rss_pred_norm = utils.get_norm(rss_pred, 1)

            utils.draw_map(aoa_pred_norm[0], f"aoa_pred")
            utils.draw_map(rss_pred_norm[0], f"rss_pred")

            joint_pred_norm = torch.cat((rss_pred_norm, aoa_pred_norm), dim=1)
            Heatmap_matrix = self.Heatmap_transform(joint_pred_norm).squeeze()

        Heatmap_matrix = (Heatmap_matrix - torch.min(Heatmap_matrix)) / (torch.max(Heatmap_matrix) - torch.min(Heatmap_matrix))

        utils.draw_map(Heatmap_matrix,f"Heatmap_matrix")
        significant_coords = []
        value_list = []
        if mode == "LocUNet_softargmax" or mode == "proposal" :
            softargmax = SoftArgmax(map_size, map_size)
            significant_coords = softargmax(Heatmap_matrix.unsqueeze(0).unsqueeze(1))

            for x, y in significant_coords:
                value_list.append(Heatmap_matrix[x.round().long(), y.round().long()])
        elif mode == "LocUNet":
            dsnt = DSNT(map_size, map_size)
            temp = Heatmap_matrix.unsqueeze(0).unsqueeze(1) / torch.sum(Heatmap_matrix.unsqueeze(0).unsqueeze(1))
            significant_coords = dsnt(temp)

            for x, y in significant_coords:
                value_list.append(Heatmap_matrix[x.round().long(), y.round().long()])
        elif mode == "DCAe" or mode == "DCAe_simple":
            coords = utils.find_maxarg_single(Heatmap_matrix.unsqueeze(0).unsqueeze(1))
            significant_coords.append(coords)
            for x, y in significant_coords:
                value_list.append(Heatmap_matrix[x, y])
        else:
            significant_coords, value_list = utils.find_maxarg_in_regions(Heatmap_matrix.unsqueeze(0).unsqueeze(1))

        for loss, S_pred in zip(significant_coords, value_list):
            print(f"Loss: {loss}, S_pred: {S_pred}")

        return significant_coords, value_list
import torch
import torch.nn as nn
import torch.optim as optim
import clip
from nerf_model import NeRFModel
from mappers import ShapeMapper, ColorMapper,init_mappers
from render import render_rays,get_rays
import numpy as np
import torch.nn.init as init
import torch.nn.functional as F

def init_nerf_weights(model):
    """
    随机初始化 NeRF 模型的权重。
    """
    for m in model.modules():
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)


def random_camera_pose():
    """
    随机选择相机的位置和朝向。
    """
    # 假设场景在单位立方体内，随机选择相机位置和朝向
    camera_pos = torch.rand(3, dtype=torch.float32)  # 随机位置
    look_at = torch.rand(3, dtype=torch.float32)  # 随机观察目标
    up = torch.tensor([0, 1, 0], dtype=torch.float32)  # 上方向

    # 计算相机朝向
    z_axis = F.normalize(look_at - camera_pos, dim=-1)
    x_axis = F.normalize(torch.cross(z_axis, up), dim=-1)
    y_axis = torch.cross(x_axis, z_axis)

    # 构造相机到世界的变换矩阵
    c2w = torch.stack([x_axis, y_axis, z_axis, camera_pos], dim=-1)
    c2w = torch.cat([c2w, torch.tensor([[0, 0, 0, 1]], dtype=torch.float32)], dim=0)  # 添加最后一行 [0, 0, 0, 1]

    return c2w

def get_camera_params(fov, H, W):
    """
    根据给定的视场角（fov）和图像尺寸（H, W），计算相机的内参矩阵K。
    """
    focal = 0.5 * W / np.tan(0.5 * fov)
    K = torch.tensor([[focal, 0, 0.5 * W],
                      [0, focal, 0.5 * H],
                      [0, 0, 1]])
    return K

# 初始化CLIP模型
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# 初始化NeRF模型
nerf_model = NeRFModel()
init_nerf_weights(nerf_model)
c2w = random_camera_pose()

H, W = 64,64
fov = 45
K = get_camera_params(fov,H,W)


rays_o, rays_d = get_rays(H,W,K,c2w)
shape_code, color_code = init_mappers(64,32)
rendered_image = render_rays((rays_o,rays_d), nerf_model,shape_code,color_code,N_samples=8)

import matplotlib.pyplot as plt
plt.imshow(rendered_image.cpu().numpy())
plt.axis('off')
plt.show()

#
# # 初始化Shape Mapper和Color Mapper
# shape_mapper = ShapeMapper()
# color_mapper = ColorMapper()
#
# # 定义优化器
# optimizer = optim.Adam(nerf_model.parameters(), lr=1e-4)
#
# # 加载数据集
# # 这里需要根据你的数据集情况进行加载和预处理
#
# num_epochs = 1000
# # 训练循环
# for epoch in range(num_epochs):
#     for text_description, image_data in dataset:
#         text_embedding = preprocess(text_description).to(device)
#         image_embedding = preprocess(image_data).to(device)
#
#         # 计算Shape Code和Color Code
#         shape_code = shape_mapper(text_embedding)
#         color_code = color_mapper(image_embedding)
#
#         # 生成NeRF的密度和颜色
#         density_sigma = nerf_model(shape_code)
#         color_c = nerf_model(color_code)
#
#         # 渲染图像
#         rendered_image = render_image(density_sigma, color_c)
#
#         # 计算图像与文本描述的相似度
#         text_embedding = text_embedding.unsqueeze(0)  # 添加batch维度
#         similarity = clip_model(text_embedding, rendered_image.unsqueeze(0))
#
#         # 计算损失
#         loss = 1 - similarity.mean()  # 这里可以根据需要调整损失函数
#
#         # 反向传播和优化
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         # 打印训练信息
#         print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
#
# # 保存模型参数
# torch.save(nerf_model.state_dict(), "nerf_model.pth")

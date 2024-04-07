#shape mapper 和 Color mapper
import torch
import torch.nn as nn
import torch.nn.functional as F


class ShapeMapper(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ShapeMapper, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2,inplace=False),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2,inplace=False),
            nn.Linear(256, output_dim)
        ).float()

    def forward(self, x):
        return self.network(x)


class ColorMapper(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ColorMapper, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2,inplace=False),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2,inplace=False),
            nn.Linear(256, output_dim)
        ).float()

    def forward(self, x):
        return self.network(x)

def init_mappers(shape_dim, color_dim):


    # 初始化 Shape Mapper 和 Color Mapper
    shape_mapper = ShapeMapper(input_dim=512, output_dim=shape_dim)
    color_mapper = ColorMapper(input_dim=512, output_dim=color_dim)

    # 随机生成输入向量（假设输入维度为 shape_dim 和 color_dim）
    input_shape = torch.randn(512)
    input_color = torch.randn(512)

    # 通过 Shape Mapper 和 Color Mapper 进行映射
    shape_code = shape_mapper(input_shape)
    color_code = color_mapper(input_color)
    return shape_code, color_code

# shape_offset, color_offset = init_mappers(63,27)
# print(shape_offset.size(), color_offset.size())
import torch
import torch.nn as nn
import torch.nn.functional as F
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4],use_viewdirs=False):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.output_ch = output_ch
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        # self.shape_dim = shape_dim
        # self.color_dim = color_dim
        device = device = "cuda" if torch.cuda.is_available() else "cpu"
        # 定义位置编码到特征的MLP 其中第0层MLP 输入 编码后的位置信息和Shape_code, 第4层的MLP 输入上一层的输出 + 编码后的位置信息+color_code
        self.pts_linears = nn.ModuleList([nn.Linear(input_ch, W)] +
                                         [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch , W) for i
                                          in range(D - 1)])
        # # 将线性层移到指定设备上
        # for l in self.pts_linears:
        #     l.to(device)

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)]) 
            self.rgb_linear = nn.Linear(W // 2, 3)
            # 将线性层移到指定设备上
            # self.feature_linear.to(device)
            # self.alpha_linear.to(device)
            # for l in self.views_linears:
            #     l.to(device)
            # self.rgb_linear.to(device)
        else:
            self.output_linear = nn.Linear(W, output_ch)
            # self.output_linear.to(device)

    def forward(self, x):

        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts 


        # 遍历位置编码的MLP
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)  
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1) 

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h) 
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs



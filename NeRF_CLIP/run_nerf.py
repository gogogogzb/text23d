import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional
import torch.nn.functional
import torch.nn.functional
from tqdm import tqdm, trange
import pytorch_msssim.ssim as SSIM
import matplotlib.pyplot as plt
from run_nerf_helpers import *
import math
from mappers import ShapeMapper, ColorMapper
import clip
from PIL import Image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False

def init_mappers(shape_dim, color_dim):
    # 初始化 Shape Mapper 和 Color Mapper
    shape_mapper = ShapeMapper(input_dim=512, output_dim=shape_dim).to(device=device)
    color_mapper = ColorMapper(input_dim=512, output_dim=color_dim).to(device=device)
    return shape_mapper, color_mapper


def get_code(text, clip_model,shape_mapper, color_mapper): 
    """
    text [batchsize , text size]
    """
    with torch.no_grad():
        tokenized_text = clip.tokenize(text).to(device)
        text_embedding =clip_model.encode_text(tokenized_text).float() #[batchsize , 512]
    # print("text size ",text_embedding.size())
    text_embedding = text_embedding.to(device)
    print("text embedding size ",text_embedding.size())
    shape_code = shape_mapper(text_embedding)
    color_code = color_mapper(text_embedding)
    # print("shape code size ",shape_code.size())
    return shape_code, color_code,text_embedding #[batchsize, 64]


def get_code2(text, clip_model,shape_mapper, color_mapper,N_rays, N_samples): 
    """
    text [batchsize , text size]
    """
    with torch.no_grad():
        tokenized_text = clip.tokenize(text).to(device)
        text_embedding =clip_model.encode_text(tokenized_text).float() #[batchsize , 512]
    # print("text size ",text_embedding.size())
    text_embedding = text_embedding.to(device)
    shape_code = shape_mapper(text_embedding)
    color_code = color_mapper(text_embedding)
    shape_code = shape_code.repeat(N_rays, N_samples, 1).reshape(-1, shape_code.shape[-1])  # [N_rays * N_samples, shape_dim]
    color_code = color_code.repeat(N_rays, N_samples, 1).reshape(-1, color_code.shape[-1])  # [N_rays * N_samples, color_dim]
    # print("shape code size ",shape_code.size())
    return shape_code, color_code,text_embedding

def get_select_inds(N_samples, iterations, random_scale=True, random_shift=True):
    N_samples_sqrt = int(np.sqrt(N_samples))
    w, h = torch.meshgrid([torch.linspace(-1, 1, N_samples_sqrt),
                                     torch.linspace(-1, 1, N_samples_sqrt)])
    h = h.unsqueeze(2)
    w = w.unsqueeze(2)

    scale_anneal = 0.0025
    min_scale = 0.2
    max_scale = 1.0
    if scale_anneal > 0:
        k_iter = iterations // 1000 * 3
        min_scale = max(min_scale, max_scale * np.exp(-k_iter * scale_anneal))
        min_scale = min(0.9, min_scale)
    else:
        min_scale = 0.25

    scale = 1
    if random_scale:
        scale = torch.Tensor(1).uniform_(min_scale, max_scale)
        h = h * scale
        w = w * scale

    if random_shift:
        max_offset = 1 - scale.item()
        h_offset = torch.Tensor(1).uniform_(0, max_offset) * (torch.randint(2, (1,)).float() - 0.5) * 2
        w_offset = torch.Tensor(1).uniform_(0, max_offset) * (torch.randint(2, (1,)).float() - 0.5) * 2

        h += h_offset
        w += w_offset

    return torch.cat([h, w], dim=2) 


def get_select_inds2(N_samples, iterations, random_scale=False, random_shift=False):
    N_samples_sqrt = int(np.sqrt(N_samples))
    w, h = torch.meshgrid([torch.linspace(-1, 1, N_samples_sqrt),
                           torch.linspace(-1, 1, N_samples_sqrt)])
    h = h.unsqueeze(2)
    w = w.unsqueeze(2)

    min_scale = 0.9  # 设置为靠近1的值，减少缩放的影响
    max_scale = 1.5  # 稍微允许一些缩放，但不是太多

    scale = 1
    if random_scale:
        # 使用固定的缩放范围来保持坐标的均匀性
        scale = torch.Tensor(1).uniform_(min_scale, max_scale)
        h = h * scale
        w = w * scale

    if random_shift:
        # 使用较小的偏移量，确保坐标不会过度集中
        max_offset = 0.2  # 减少偏移量的最大值，以保持坐标的均匀分布
        h_offset = torch.Tensor(1).uniform_(-max_offset, max_offset)
        w_offset = torch.Tensor(1).uniform_(-max_offset, max_offset)

        h += h_offset
        w += w_offset

    return torch.cat([h, w], dim=2)

def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    chunk 并行处理的射线的数量
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, fn,shape_code, color_code, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
     inputs 也就是pts [N_rays, N_samples, 3]
     viewdirs [N_rays, 3]
     fn :nerfmodel
     netchunk 并行处理输入网络的点的数量
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]]) #[N_rays * N_samples, 3]
    embedded = embed_fn(inputs_flat)#[N_rays * N_samples, 63] 其中63  =  2 * 10 * 3 +3

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape) #[N_rays, N_samples, 3]
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]]) #[N_rays * N_samples, 3]
        embedded_dirs = embeddirs_fn(input_dirs_flat) #[N_rays * N_samples, 27] 27 = 3 * 4*2 + 3
        embedded = torch.cat([embedded, embedded_dirs], -1) #[N_rays * N_samples, 90] 90 = 63 + 27
    # 假设shape_code和color_code是每个光线的属性，需要扩展为每个采样点
    # shape_code和color_code[ shape/color_dim , 1]
    shape_code = shape_code.transpose(0, 1).expand(inputs.shape[0], -1)  # [N_rays, shape_dim]
    color_code = color_code.transpose(0, 1).expand(inputs.shape[0], -1)  # [N_rays, color_dim]

    shape_code_expanded = shape_code[:, None, :].expand(-1, inputs.shape[1], -1).reshape(-1, shape_code.shape[-1]) #[N_rays * N_samples, shape_dim]
    color_code_expanded = color_code[:, None, :].expand(-1, inputs.shape[1], -1).reshape(-1, color_code.shape[-1]) #[N_rays * N_samples, color_dim]


    embedded = torch.cat([embedded, shape_code_expanded, color_code_expanded], -1)
    outputs_flat = batchify(fn, netchunk)(embedded) #[N_rays * N_samples, 4] 4 包括 RGb 和 alpha
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0.01, white_bkgd=False, pytest=False):

    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]
    
    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)
    # print("dist size ",dists.size())

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch,
                network_fn,
                N_samples,embed_fn,embeddirs_fn,shape_code,color_code):
    N_rays = ray_batch.shape[0]
    print("N_rays = ", N_rays)
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    z_vals = near * (1.-t_vals) + far * (t_vals)
    z_vals = z_vals.expand([N_rays, N_samples])

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]
    print("pts size = ",pts.size())
    embedded = get_embeddings(pts, viewdirs, embed_fn, embeddirs_fn)
    print("embeed size = {}, shape code size = {}".format(embedded.size(), shape_code.size()))
    embedded = torch.cat([embedded, shape_code, color_code], -1)
#     raw = run_network(pts)
    raw = network_fn(embedded)
    outputs = torch.reshape(raw, list(pts.shape[:-1]) + [raw.shape[-1]])
    print("raw size = {}",raw.size())
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(outputs, z_vals, rays_d)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def batchify_rays(rays_flat,nerf_model,N_samples,embed_fn, embeddirs_fn,shape_code,color_code, chunk=1024):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk],nerf_model,
                N_samples,embed_fn,embeddirs_fn,shape_code,color_code )
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, K,shape_code,color_code,nerf_model, N_samples, embed_fn,embeddirs_fn, c2w, chunk=1024*32, rays=None,
                  near=0., far=1.,
                  use_viewdirs=True):
    rays_o, rays_d = get_rays(H, W, K, c2w)


    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]


    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()
    print("rays size = {}".format(rays_o.size()))
    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays,nerf_model,N_samples,embed_fn, embeddirs_fn,shape_code,color_code)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i==0:
            print(rgb.shape, disp.shape)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)


    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)
    text = args.text_input
    shape_code , color_code = get_code(text)
    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,shape_code=shape_code,color_code = color_code,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer



# def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):

#     raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

#     dists = z_vals[...,1:] - z_vals[...,:-1]
#     dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

#     dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

#     rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
#     noise = 0.
#     if raw_noise_std > 0.:
#         noise = torch.randn(raw[...,3].shape) * raw_noise_std

#         # Overwrite randomly sampled data if pytest
#         if pytest:
#             np.random.seed(0)
#             noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
#             noise = torch.Tensor(noise)

#     alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
#     # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
#     weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
#     rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

#     depth_map = torch.sum(weights * z_vals, -1)
#     disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
#     acc_map = torch.sum(weights, -1)

#     if white_bkgd:
#         rgb_map = rgb_map + (1.-acc_map[...,None])

#     return rgb_map, disp_map, acc_map, weights, depth_map




def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/nerf_llff_data/fern',
                        help='input data directory')
    parser.add_argument("--text_input",type=str,default='a green hat',
                        help="the text description")
    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    #每一次梯度下降射线的数量
    parser.add_argument("--N_rand", type=int, default=32*4*4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    #可以适当降低，不然会 out of memory
    parser.add_argument("--chunk", type=int, default=128*32,
                        help='number of rays processed in parallel, decrease if running out of memory')

    parser.add_argument("--netchunk", type=int, default=128*64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=1000,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=1000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=2000,
                        help='frequency of render_poses video saving')

    return parser


def train():

    parser = config_parser()
    args = parser.parse_args()

    # Load data
    K = None
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
            
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'LINEMOD':
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir, args.half_res, args.testskip)
        print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split

        if args.white_bkgd:
            images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
        else:
            images = images[...,:3]

    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            rgbs, _ = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            return

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching
        print('get rays')
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        print('done, concats')
        rays_rgb = np.concatenate([rays, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0) # train images only
        rays_rgb = np.reshape(rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)

        print('done')
        i_batch = 0

    # Move training data to GPU
    if use_batching:
        images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)

    #训练次数
    N_iters = 20000 + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # Summary writers
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))
    
    start = start + 1
    for i in trange(start, N_iters):
        time0 = time.time()

        # Sample random ray batch
        if use_batching:
            # Random over all images
            batch = rays_rgb[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0

        else:
            # Random from one image
            img_i = np.random.choice(i_train)
            target = images[img_i]
            target = torch.Tensor(target).to(device)
            pose = poses[img_i, :3,:4]

            if N_rand is not None:
                rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                if i < args.precrop_iters:
                    dH = int(H//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                            torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                        ), -1)
                    if i == start:
                        print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

        #####  Core optimization loop  #####
        rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)

        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s)
        trans = extras['raw'][...,-1]
        loss = img_loss
        psnr = mse2psnr(img_loss)

        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        dt = time.time()-time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if i%args.i_video==0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

        if i%args.i_testset==0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
            print('Saved test set')


    
        if i%args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")

        global_step += 1





# def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
#     """Transforms model's predictions to semantically meaningful values.
#     Args:
#         raw: [num_rays, num_samples along ray, 4]. Prediction from model.
#         z_vals: [num_rays, num_samples along ray]. Integration time.
#         rays_d: [num_rays, 3]. Direction of each ray.
#     Returns:
#         rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
#         disp_map: [num_rays]. Disparity map. Inverse of depth map.
#         acc_map: [num_rays]. Sum of weights along each ray.
#         weights: [num_rays, num_samples]. Weights assigned to each sampled color.
#         depth_map: [num_rays]. Estimated distance to object.
#     """
#     raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

#     dists = z_vals[...,1:] - z_vals[...,:-1]
#     dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

#     dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

#     rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
#     noise = 0.
#     if raw_noise_std > 0.:
#         noise = torch.randn(raw[...,3].shape) * raw_noise_std

#         # Overwrite randomly sampled data if pytest
#         if pytest:
#             np.random.seed(0)
#             noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
#             noise = torch.Tensor(noise)

#     alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
#     # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
#     weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
#     rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

#     depth_map = torch.sum(weights * z_vals, -1)
#     disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
#     acc_map = torch.sum(weights, -1)

#     if white_bkgd:
#         rgb_map = rgb_map + (1.-acc_map[...,None])

#     return rgb_map, disp_map, acc_map, weights, depth_map




def get_embeddings(pts, viewdirs,embed_fn, embeddirs_fn, shape_offset, color_offset):
    inputs_flat = torch.reshape(pts, [-1, pts.shape[-1]]) #[N_rays * N_samples, 3]
    embedded = embed_fn(inputs_flat)#[N_rays * N_samples, 63] 其中63  =  2 * 10 * 3 +3
    embedded = embedded + shape_offset
    input_dirs = viewdirs[:,None].expand(pts.shape) #[N_rays, N_samples, 3]
    input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]]) #[N_rays * N_samples, 3]
    embedded_dirs = embeddirs_fn(input_dirs_flat) #[N_rays * N_samples, 27] 27 = 3 * 4*2 + 3
    embedded_dirs = embedded_dirs + color_offset
    embedded = torch.cat([embedded, embedded_dirs], -1) #[N_rays * N_samples, 90] 90 = 63 + 27
    return embedded


# def make_train(img_id):
   
#     source_path = './data/train_imgs/'
#     image_ids, captions = load_datas()
#      # Set device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     embed_fn, input_ch = get_embedder(10, 0)
#     # Set render parameters
#     N_rays = 1024
#     focal = 1     # Focal length
#     near = 1
#     far = 6
#     N_samples = 32
#     embeddirs_fn, input_ch_views = get_embedder(4, 0)
#     # Load or initialize Nerf model
#     nerf_model = NeRF(input_ch= input_ch, input_ch_views= input_ch_views,use_viewdirs=True).to(device)

#     clip_model, preprocess = clip.load("ViT-B/32", device=device)
#     shape_mapper , color_mapper = init_mappers(512,512)

#     learnable_camera_pose = LearnableCameraPose().to(device)
#     optimizer = torch.optim.Adam([
#         {'params': nerf_model.parameters()},
#         {'params': shape_mapper.parameters()},
#         {'params': color_mapper.parameters()},
#         {'params':learnable_camera_pose.parameters()}
#     ], lr=1e-3)

#     pretrained_paths = {
#     'nerf_model': r'D:\gzb\NeRF_CLIP\saved_one_img_models\one_img_nerf_model_epoch_60000.pth',
#     'shape_mapper': r'D:\gzb\NeRF_CLIP\saved_one_img_models\one_img_shape_mapper_epoch_60000.pth',
#     'color_mapper': r'D:\gzb\NeRF_CLIP\saved_one_img_models\one_img_color_mapper_epoch_60000.pth',
#     'learnable_camera_pose': r'D:\gzb\NeRF_CLIP\saved_one_img_models\one_img_learnable_camera_pose_epoch_60000.pth'
# }
    
#     models = {
#     'nerf_model': nerf_model,
#     'shape_mapper': shape_mapper,
#     'color_mapper': color_mapper,
#     'learnable_camera_pose': learnable_camera_pose
# }
# # 加载预训练状态
#     load_pretrained_models(pretrained_paths, models)
#     #相机的内参矩阵
   
#     num_epochs = 5000 
#     total_epoch = 60000 #之前训练的总轮数

#     img_dir = os.path.join(source_path,image_ids[img_id])
#     img = Image.open(img_dir)
#     img_tensor = torch.tensor(np.array(img), dtype=torch.float, device=device) /255.0
#     # print(img_tensor.size())
#     H, W = img.height, img.width
#     # print("img size = ",H,W)
#     K = np.array([ 
#         [focal, 0, 0.5*W],
#         [0, focal, 0.5*H],
#         [0, 0, 1]
#     ])
#     text = captions[img_id]
#     print(f"循环之前分配的显存: {torch.cuda.memory_allocated('cuda:0') / 1e9} GB")
#     for epoch in range(1+total_epoch,num_epochs+1+total_epoch):
#         position, orientation = learnable_camera_pose()
#         rotation_matrix = R.from_quat(orientation.cpu().detach().numpy()).as_matrix()
#         rotation_matrix = torch.tensor(rotation_matrix, dtype=torch.float32).to(device)
#         c2w = torch.eye(4, device=device)
#         c2w[:3, :3] = rotation_matrix
#         c2w[:3, 3] = position
#         shape_code, color_code, text_embedding = get_code2(text, clip_model, shape_mapper, color_mapper,N_rays,N_samples)
#         render_img = torch.zeros_like(img_tensor)
#         render(H,W,K,shape_code,color_code,nerf_model,N_samples,embed_fn,embeddirs_fn,c2w)
#         # print(f"循环开始分配的显存: {torch.cuda.memory_allocated('cuda:0') / 1e9} GB")
#         # start_time = time.time()
#         # total_loss = 0.0
       
#         # endtime =time.time()
#         # print("epoch {} use time = {} s".format(epoch,endtime - start_time))
#         # total_loss = F.mse_loss(render_img,img_tensor)
#         # print("mse loss = ",total_loss)
#         # print("render img size = ",render_img.size())
#         # render_img = render_img.permute(2, 0, 1)
#         # import torchvision.transforms.functional as fun
#         # render_img_resized = fun.resize(render_img,[224,224])
#         # # 确保尺寸和通道数正确
#         # render_img_resized = render_img_resized.permute(1, 2, 0)

#         # # 添加批次维度，形状变为 [1, 224, 224, 3]
#         # render_img_resized = render_img_resized.unsqueeze(0)

#         # # 重新将通道移到第二个维度，形状变为 [1, 3, 224, 224]
#         # render_img_resized = render_img_resized.permute(0, 3, 1, 2)
#         # render_img_feature = clip_model.encode_image(render_img_resized)
#         # similarity = torch.cosine_similarity(text_embedding,render_img_feature)
#         # clip_loss = 1 - similarity
#         # total_loss = total_loss + clip_loss
#         # print("epoch {}, loss = {}".format(epoch, total_loss))
        
#         # optimizer.zero_grad()
#         # total_loss.backward()
#         # optimizer.step()
        
#         # if epoch %10000==0:
#         #     save_one_img_models(epoch,models,optimizer)



import torchvision.transforms.functional as TF
def compute_mse_loss(predicted_image, target_image):
    # Compute MSE loss
    target_image_np = TF.to_tensor(target_image).numpy()
    predicted_image_np = TF.to_tensor(predicted_image).numpy()
    predicted_image = torch.tensor(predicted_image_np).float()
    target_image =  torch.tensor(target_image_np).float()
    mse_loss = torch.mean(torch.square(predicted_image - target_image))
    return mse_loss

from load_data import load_datas

from torch.autograd import profiler


        
def load_pretrained_models(pretrained_paths, models):
    for model_name, path in pretrained_paths.items():
        if os.path.exists(path):
            models[model_name].load_state_dict(torch.load(path))
            print(f"Loaded pretrained weights for {model_name} from {path}")
        else:
            print(f"No pretrained weights found for {model_name} at {path}")


def save_models(epoch, models, optimizer, save_dir='./saved_models/'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for model_name, model in models.items():
        torch.save(model.state_dict(), os.path.join(save_dir, f'one_img_{model_name}_epoch_{epoch}.pth'))
    torch.save(optimizer.state_dict(), os.path.join(save_dir, f'ong_img_optimizer_epoch_{epoch}.pth'))
    print(f"Models and optimizer states have been saved for epoch {epoch}.")


def save_one_img_models(epoch, models, optimizer, save_dir='./saved_one_img_models/'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for model_name, model in models.items():
        torch.save(model.state_dict(), os.path.join(save_dir, f'multi_view_img_{model_name}_epoch_{epoch}.pth'))
    torch.save(optimizer.state_dict(), os.path.join(save_dir, f'multi_view_img_optimizer_epoch_{epoch}.pth'))
    print(f"Models and optimizer states have been saved for epoch {epoch}.")
from scipy.spatial.transform import Rotation as R




def try_train():
   
    source_path = './data/new_imgs/'
    image_ids, captions = load_datas()
     # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_fn, input_ch = get_embedder(10, 0)
    # Set render parameters
    N_rays = 1024
    focal = 1     # Focal length
    near = 1
    far = 6
    N_samples = 64
    embeddirs_fn, input_ch_views = get_embedder(4, 0)
    # Load or initialize Nerf model
    nerf_model = NeRF(input_ch= input_ch, input_ch_views= input_ch_views,use_viewdirs=True).to(device)

    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    shape_mapper , color_mapper = init_mappers(512,512)

    learnable_camera_pose = LearnableCameraPose().to(device)
    optimizer = torch.optim.Adam([
        {'params': nerf_model.parameters()},
        {'params': shape_mapper.parameters()},
        {'params': color_mapper.parameters()},
        {'params':learnable_camera_pose.parameters()}
    ], lr=1e-3)

    
    pretrained_paths = {
    'nerf_model': r'E:\Pythoncode\NeRF_CLIP\saved_models\nerf_model_epoch_35000.pth',
    'shape_mapper': r'E:\Pythoncode\NeRF_CLIP\saved_models\shape_mapper_epoch_35000.pth',
    'color_mapper': r'E:\Pythoncode\NeRF_CLIP\saved_models\color_mapper_epoch_35000.pth',
    'learnable_camera_pose': r'E:\Pythoncode\NeRF_CLIP\saved_models\learnable_camera_pose_epoch_35000.pth'
}
    models = {
    'nerf_model': nerf_model,
    'shape_mapper': shape_mapper,
    'color_mapper': color_mapper,
    'learnable_camera_pose': learnable_camera_pose
}
    # total_epoch = 12000
# 加载预训练状态
    load_pretrained_models(pretrained_paths, models)
    #相机的内参矩阵
   
    num_epochs = 5000
    total_epoch = 35000

    #针对所有图像进行训练
    for i in range(len(image_ids)):
        img_dir = os.path.join(source_path,image_ids[i])
        img = Image.open(img_dir)
        img_tensor = torch.tensor(np.array(img), dtype=torch.float, device=device)
        img_tensor = img_tensor /255.0
        img_tensor = img_tensor.reshape(-1,3)
        H, W = img.height, img.width
        K = np.array([ 
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])
        # print(" img size  ={} * {} ".format(H ,W))
        text = captions[i]


        for epoch in range(1 + i *num_epochs + total_epoch,1 + i*num_epochs +num_epochs + total_epoch):
            position, orientation = learnable_camera_pose()
            rotation_matrix = R.from_quat(orientation.cpu().detach().numpy()).as_matrix()
            rotation_matrix = torch.tensor(rotation_matrix, dtype=torch.float32).to(device)
            c2w = torch.eye(4, device=device)
            c2w[:3, :3] = rotation_matrix
            c2w[:3, 3] = position
        
            t_vals = torch.linspace(0., 1., steps=N_samples)
            z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))
            z_vals = z_vals.expand([N_rays, N_samples])
            rays_o, rays_d = get_rays(H, W, K, c2w) #[H, W, 3]
            # print("ray o ",rays_o.size())
            rays_o = rays_o.reshape(-1, 3) #[H * W , 3 ]
            rays_d = rays_d.reshape(-1, 3) #[H * W , 3 ]

            shape_code0, color_code0, text_embedding = get_code(text, clip_model, shape_mapper, color_mapper)
            shape_code = shape_code0.repeat(N_rays, 1)  # [N_rays, shape_dim]
            color_code = color_code0.repeat(N_rays, 1)  # [N_rays, color_dim]
            shape_code = shape_code[:, None, :].expand(-1, N_samples, -1).reshape(-1, shape_code.shape[
                -1])  # [N_rays * N_samples, shape_dim]
            color_code = color_code[:, None, :].expand(-1, N_samples, -1).reshape(-1, color_code.shape[
                -1])  # [N_rays * N_samples, shape_dim]
            # rgb = torch.empty((H,W,3),dtype=torch.float)
            # print("image {} epoch {}".format(i, epoch))
            selected_indices = torch.randint(0, H * W, (N_rays,))
            img_block = img_tensor[selected_indices]
            # print("img_block size = ", img_block.size())
            selected_rays_o = rays_o[selected_indices]
            selected_rays_d = rays_d[selected_indices]
            viewdirs = selected_rays_d
            viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
            viewdirs = torch.reshape(viewdirs, [-1, 3]).float()
            pts = selected_rays_o[..., None, :] + selected_rays_d[..., None, :] * z_vals[..., :,None]  # None 相当于添加1维度， 如rays_o[N_rays，3] -> [N_rays,1,3]

            embedded = get_embeddings(pts, viewdirs, embed_fn, embeddirs_fn)
            embedded = torch.cat([embedded, shape_code, color_code], -1)
            # with torch.autograd.set_detect_anomaly(True):
            raw = nerf_model(embedded)
            outputs = torch.reshape(raw, list(pts.shape[:-1]) + [raw.shape[-1]])
            # print("outputs size = ", outputs.size())
            rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(outputs, z_vals, selected_rays_d)
            # print("rgb_map size ", rgb_map.size())
            # img_block = img_tensor[i * N_rays: (i + 1) * N_rays]
            # print("img_block size = ",img_block.size())
            # print("compare ",rgb_map[0], img_block[0])
            mse_loss = F.mse_loss(rgb_map, img_block)
            print("epoch = {}, loss = {}".format(epoch,mse_loss))
            # total_mse_loss += mse_loss
            # print("total loss = ", total_mse_loss)
            optimizer.zero_grad()
            mse_loss.backward()
            optimizer.step()
            if epoch %10000==0:
                save_models(epoch,models,optimizer)
         


def one_img_train(img_id):
   
    source_path = './data/new_imgs/'
    image_ids, captions = load_datas()
     # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_fn, input_ch = get_embedder(10, 0)
    # Set render parameters
    N_rays = 1024
    focal = 1     # Focal length
    near = 1
    far = 6
    N_samples = 64
    embeddirs_fn, input_ch_views = get_embedder(4, 0)
    # Load or initialize Nerf model
    nerf_model = NeRF(input_ch= input_ch, input_ch_views= input_ch_views,use_viewdirs=True).to(device)

    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    shape_mapper , color_mapper = init_mappers(512,512)

    learnable_camera_pose = LearnableCameraPose().to(device)
    optimizer = torch.optim.Adam([
        {'params': nerf_model.parameters()},
        {'params': shape_mapper.parameters()},
        {'params': color_mapper.parameters()},
        {'params':learnable_camera_pose.parameters()}
    ], lr=1e-3)

    pretrained_paths = {
    'nerf_model': r'E:\Pythoncode\NeRF_CLIP\saved_models\one_img_nerf_model_epoch_60000.pth',
    'shape_mapper': r'E:\Pythoncode\NeRF_CLIP\saved_models\one_img_shape_mapper_epoch_60000.pth',
    'color_mapper': r'E:\Pythoncode\NeRF_CLIP\saved_models\one_img_color_mapper_epoch_60000.pth',
    'learnable_camera_pose': r'E:\Pythoncode\NeRF_CLIP\saved_models\one_img_learnable_camera_pose_epoch_60000.pth'
}
    
    models = {
    'nerf_model': nerf_model,
    'shape_mapper': shape_mapper,
    'color_mapper': color_mapper,
    'learnable_camera_pose': learnable_camera_pose
}
# 加载预训练状态
    load_pretrained_models(pretrained_paths, models)
    #相机的内参矩阵
   
    num_epochs = 5000 
    total_epoch = 60000 #之前训练的总轮数

    img_dir = os.path.join(source_path,image_ids[img_id])
    img = Image.open(img_dir)
    img_tensor = torch.tensor(np.array(img), dtype=torch.float, device=device)
    img_tensor = img_tensor /255.0
    img_tensor = img_tensor.reshape(-1,3)
    H, W = img.height, img.width
    K = np.array([ 
        [focal, 0, 0.5*W],
        [0, focal, 0.5*H],
        [0, 0, 1]
    ])
    text = captions[img_id]

    for epoch in range(1+total_epoch,num_epochs+1+total_epoch):
        position, orientation = learnable_camera_pose()
        rotation_matrix = R.from_quat(orientation.cpu().detach().numpy()).as_matrix()
        rotation_matrix = torch.tensor(rotation_matrix, dtype=torch.float32).to(device)
        c2w = torch.eye(4, device=device)
        c2w[:3, :3] = rotation_matrix
        c2w[:3, 3] = position
    
        t_vals = torch.linspace(0., 1., steps=N_samples)
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))
        z_vals = z_vals.expand([N_rays, N_samples])
        rays_o, rays_d = get_rays(H, W, K, c2w) #[H, W, 3]
        # print("ray o ",rays_o.size())
        rays_o = rays_o.reshape(-1, 3) #[H * W , 3 ]
        rays_d = rays_d.reshape(-1, 3) #[H * W , 3 ]

        shape_code0, color_code0, text_embedding = get_code(text, clip_model, shape_mapper, color_mapper)
        shape_code = shape_code0.repeat(N_rays, 1)  # [N_rays, shape_dim]
        color_code = color_code0.repeat(N_rays, 1)  # [N_rays, color_dim]
        shape_code = shape_code[:, None, :].expand(-1, N_samples, -1).reshape(-1, shape_code.shape[
            -1])  # [N_rays * N_samples, shape_dim]
        color_code = color_code[:, None, :].expand(-1, N_samples, -1).reshape(-1, color_code.shape[
            -1])  # [N_rays * N_samples, shape_dim]

        selected_indices = torch.randint(0, H * W, (N_rays,))
        img_block = img_tensor[selected_indices]
        # print("img_block size = ", img_block.size())
        selected_rays_o = rays_o[selected_indices]
        selected_rays_d = rays_d[selected_indices]
        viewdirs = selected_rays_d
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()
        pts = selected_rays_o[..., None, :] + selected_rays_d[..., None, :] * z_vals[..., :,None]  # None 相当于添加1维度， 如rays_o[N_rays，3] -> [N_rays,1,3]

        embedded = get_embeddings(pts, viewdirs, embed_fn, embeddirs_fn)
        embedded = torch.cat([embedded, shape_code, color_code], -1)
        raw = nerf_model(embedded)
        outputs = torch.reshape(raw, list(pts.shape[:-1]) + [raw.shape[-1]])
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(outputs, z_vals, selected_rays_d)
        mse_loss = F.mse_loss(rgb_map, img_block)
        print("epoch = {}, loss = {}".format(epoch,mse_loss))
        optimizer.zero_grad()
        mse_loss.backward()
        optimizer.step()
        if epoch %10000==0:
            save_one_img_models(epoch,models,optimizer)




def compute_clip_loss(rendered_image, text, clip_model, preprocess):
    # 将渲染的图像转换为CLIP模型所需的格式
    rendered_image_clip = preprocess(rendered_image).unsqueeze(0).to(device)
    
    # 获取文本嵌入
    text_tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
    
    # 获取图像特征并计算相似性损失
    image_features = clip_model.encode_image(rendered_image_clip)
    similarity = torch.cosine_similarity(text_features, image_features)
    clip_loss = 1 - similarity  # 相似性越高，损失越低

    return clip_loss



def generate_new_views(learnable_camera_pose, num_views, device):
    # 获取训练好的相机位置和方向
    trained_position, trained_orientation = learnable_camera_pose()
    trained_rotation_matrix =  torch.tensor(R.from_quat(trained_orientation.detach().cpu().numpy()).as_matrix(),dtype=torch.float32).to(device)
    

    new_c2w_matrices = []
    for i in range(num_views):
        # 生成新的相机方向（例如，围绕 y 轴旋转）
        angle = 2 * np.pi * i / num_views
        rotation_matrix = torch.tensor([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]],
        # rotation_matrix = torch.tensor([
        # [np.cos(angle), -np.sin(angle), 0],
        # [np.sin(angle), np.cos(angle), 0],
        # [0, 0, 1]],
         dtype=torch.float32, device=device)


        # 结合训练好的相机位置和新的相机方向构建 c2w 矩阵
        c2w = torch.eye(4, device=device)
        c2w[:3, :3] = torch.matmul(rotation_matrix, trained_rotation_matrix)
        c2w[:3, 3] = trained_position
        new_c2w_matrices.append(c2w)

    return new_c2w_matrices

def one_img_train2(): #随机选取一小块区域 而不是随机选取一些射线
   
    source_path = './data/multi_view_imgs/'
    image_ids, captions = load_datas('./data/multi_view_imgs/captions.txt')
     # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_fn, input_ch = get_embedder(10, 0)
    # Set render parameters
    N_rays = 3136
    sample_scale = 56
    focal = 1     # Focal length
    near = 1
    far = 2
    N_samples = 32
    embeddirs_fn, input_ch_views = get_embedder(4, 0)
    # Load or initialize Nerf model
    nerf_model = NeRF(input_ch= input_ch, input_ch_views= input_ch_views,use_viewdirs=True).to(device)

    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    shape_mapper , color_mapper = init_mappers(input_ch, input_ch_views)

    learnable_camera_pose = LearnableCameraPose().to(device)
    optimizer = torch.optim.Adam([
        {'params': nerf_model.parameters()},
        {'params': shape_mapper.parameters()},
        {'params': color_mapper.parameters()},
        {'params':learnable_camera_pose.parameters()}
    ], lr=1e-5)

    pretrained_paths = {
    'nerf_model': './day47/multi_view_img_nerf_model_epoch_42000.pth',
    'shape_mapper': './day47/multi_view_img_shape_mapper_epoch_42000.pth',
    'color_mapper':'./day47/multi_view_img_color_mapper_epoch_42000.pth',
    'learnable_camera_pose':'./day47/multi_view_img_learnable_camera_pose_epoch_42000.pth'
}
    
    models = {
    'nerf_model': nerf_model,
    'shape_mapper': shape_mapper,
    'color_mapper': color_mapper,
    'learnable_camera_pose': learnable_camera_pose
}
# 加载预训练状态
    load_pretrained_models(pretrained_paths, models)
    #相机的内参矩阵
   
    num_epochs = 5000
    total_num_of_epoch = 100
    sum_of_num=42000
    for iter in range(total_num_of_epoch):
        for id in range(len(image_ids)):
            img_dir = os.path.join(source_path,image_ids[id])
            img = Image.open(img_dir)
            img_tensor = torch.tensor(np.array(img), dtype=torch.float, device=device) /255.0
            # img_tensor = img_tensor.reshape(-1,3)
            # print(img_tensor.size())
            H, W = img.height, img.width
            # print("img size = ",H,W)
            K = np.array([ 
                [focal, 0, 0.5*W],
                [0, focal, 0.5*H],
                [0, 0, 1]
            ])
            text = captions[id]
            # print(f"循环之前分配的显存: {torch.cuda.memory_allocated('cuda:0') / 1e9} GB")
            for epoch in range(1,num_epochs+1):
                sum_of_num+=1
                position, orientation = learnable_camera_pose()
                rotation_matrix = R.from_quat(orientation.cpu().detach().numpy()).as_matrix()
                rotation_matrix = torch.tensor(rotation_matrix, dtype=torch.float32).to(device)
                c2w = torch.eye(4, device=device)
                c2w[:3, :3] = rotation_matrix
                c2w[:3, 3] = position
                del position,rotation_matrix
                torch.cuda.empty_cache()
                t_vals = torch.linspace(0., 1., steps=N_samples)
                z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))
                z_vals = z_vals.expand([N_rays, N_samples])
                rays_o, rays_d = get_rays(H, W, K, c2w) #[H, W, 3]


                #一种选取方法
                select_inds = get_select_inds2(sample_scale * sample_scale, sum_of_num)
                # print("ray o ",rays_o.size())
                rays_o = torch.nn.functional.grid_sample(rays_o.permute(2, 0, 1).unsqueeze(0),
                                                     select_inds.unsqueeze(0), mode='bilinear', align_corners=True)[0]
                rays_d = torch.nn.functional.grid_sample(rays_d.permute(2, 0, 1).unsqueeze(0),
                                                        select_inds.unsqueeze(0), mode='bilinear', align_corners=True)[0]
                rays_o = rays_o.permute(1, 2, 0).view(-1, 3)
                rays_d = rays_d.permute(1, 2, 0).view(-1, 3)
                # print("rays size = ",rays_o.size())
                # rays_o = rays_o.reshape(-1, 3) #[H * W , 3 ]
                # rays_d = rays_d.reshape(-1, 3) #[H * W , 3 ]
                target_s = torch.nn.functional.grid_sample(img_tensor.permute(2, 0, 1).unsqueeze(0),
                                                     select_inds.unsqueeze(0), mode='bilinear', align_corners=True)[0]
                target_s = target_s.permute(1, 2, 0).view(-1, 3)
                shape_offset, color_offset, text_embedding = get_code(text, clip_model, shape_mapper, color_mapper)

                viewdirs = rays_d
                viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
                viewdirs = torch.reshape(viewdirs, [-1, 3]).float()
                pts = rays_o[...,None,:] + rays_d[...,None,:]*z_vals[...,:,None]
                embedded = get_embeddings(pts, viewdirs, embed_fn, embeddirs_fn,shape_offset, color_offset)
                
                # embedded = torch.cat([embedded, shape_code, color_code], -1)
                # print("embeed size = {}, shape code size = {}".format(embedded.size(), shape_code.size()))
                raw = nerf_model(embedded)
                outputs = torch.reshape(raw, list(pts.shape[:-1]) + [raw.shape[-1]])
                rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(outputs, z_vals, rays_d)
                gen_img = rgb_map
                
                optimizer.zero_grad()
                
                mse_loss = F.mse_loss(gen_img,target_s)
                loss = mse_loss
                # print("rgb_map size ",rgb_map.size()) 
                rendered_img = rgb_map.view(sample_scale,sample_scale,-1).permute(2,0,1).unsqueeze(0)
                target_s = target_s.view(sample_scale,sample_scale,-1).permute(2,0,1).unsqueeze(0)
                ssim_loss = 1 - SSIM(rendered_img,target_s,data_range=1, size_average=True)
                # print("rendered img size ",img1.size())
                scale_img = F.upsample_bilinear(rendered_img,(224,224))
                # print("scale_imgs size ",scale_img.size())
                img_feature = clip_model.encode_image(scale_img)
                similarity =  torch.cosine_similarity(img_feature,text_embedding)
                clip_loss = (1 -similarity )
                print("img {} , epoch {}, similarity = {}, mse loss = {} , ssim_loss = {}".format(id+1, sum_of_num,similarity.item(),mse_loss.item(), ssim_loss.item()))
                
                loss = loss + clip_loss + ssim_loss 
                # print("img {}  epoch {}  total_loss = {} ".format(id,sum_of_num,loss))
                loss.backward()
                optimizer.step()
                # print(" 释放显存后分配显存 为 {} GB".format(torch.cuda.memory_allocated('cuda:0') / 1e9))
                # endtime =time.time()
                # print("epoch {} use time = {} s".format(sum_of_num,endtime - start_time))
                
                if epoch%500==0:
                   
                    # rays_o1, rays_d1 = get_rays(H, W, K, c2w) #[H, W, 3]
                    # rays_o1 = rays_o1.reshape(-1, 3) #[H * W , 3 ]
                    # rays_d1 = rays_d1.reshape(-1, 3) #[H * W , 3 ]
                    # rgb = torch.zeros((H,W,3),dtype=torch.float32)
                    # with torch.no_grad():
                    #     for row in range(H):
                    #         t_vals0 = torch.linspace(0., 1., steps=N_samples)
                    #         z_vals0 = 1./(1./near * (1.-t_vals0) + 1./far * (t_vals0))
                    #         z_vals0 = z_vals0.expand([H, N_samples])
                    #         selected_rays_o, seleted_rays_d = rays_o1[row * H: (row + 1) * H], rays_d1[row * H: (row+ 1) * H]
                    #         viewdirs0 = seleted_rays_d
                    #         viewdirs0 = viewdirs0 / torch.norm(viewdirs0, dim=-1, keepdim=True)
                    #         viewdirs0 = torch.reshape(viewdirs0, [-1, 3]).float()
                    #         pts0 = selected_rays_o[..., None, :] + seleted_rays_d[..., None, :] * z_vals0[..., :,
                    #                                                                                 None]  # None 相当于添加1维度， 如rays_o[N_rays，3] -> [N_rays,1,3]
                    #         # #                          [N_rays ,1,3] *[N_rays, N_samples, 1] = [N_rays, N_Samples,3]
                    #         shape_code0, color_code0, text_embedding0 = get_code2(text, clip_model, shape_mapper, color_mapper,H,N_samples)
                    #         embedded0 = get_embeddings(pts0.detach(), viewdirs0, embed_fn, embeddirs_fn,shape_offset, color_offset)
                    #         # embedded0 = torch.cat([embedded0, shape_code0, color_code0], -1)
                    #         raw0 = nerf_model(embedded0)
                    #         outputs0 = torch.reshape(raw0, list(pts0.shape[:-1]) + [raw0.shape[-1]])
                    #         rgb_map0, disp_map, acc_map, weights, depth_map = raw2outputs(outputs0, z_vals0, seleted_rays_d)
                    #         rgb_map1 = rgb_map0.detach()
                    #         # print("rgb_map size ",rgb_map0.size())
                    #         rgb[row] = rgb_map1
                    #     rgb = (rgb*255.0).to(torch.uint8).cpu()
                    save_img = rgb_map.view(sample_scale,sample_scale,-1).detach().cpu().numpy()
                    # plt.imsave(f'./data/hot_dogs/batch{iter}_epoch{epoch}_img{id + 1}.jpg',rgb.numpy())
                    plt.imsave(f'./data/hot_dogs/day47_batch{iter}_epoch{epoch}_img{id + 1}.jpg',save_img)
            save_one_img_models(sum_of_num,models,optimizer,save_dir='./day47/')              
                


def generate_multi_perspective_imgs(H,W,text):
     # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_fn, input_ch = get_embedder(10, 0)
    # Set render parameters
    N_rays = 1024
    focal = 1     # Focal length
    near = 1
    far = 2
    N_samples = 48
    embeddirs_fn, input_ch_views = get_embedder(4, 0)
    # Load or initialize Nerf model
    nerf_model = NeRF(input_ch= input_ch, input_ch_views= input_ch_views,use_viewdirs=True).to(device)

    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    # shape_mapper , color_mapper = init_mappers(512,)
    shape_mapper , color_mapper = init_mappers(input_ch, input_ch_views)
    learnable_camera_pose = LearnableCameraPose().to(device)
    optimizer = torch.optim.Adam([
        {'params': nerf_model.parameters()},
        {'params': shape_mapper.parameters()},
        {'params': color_mapper.parameters()},
        {'params':learnable_camera_pose.parameters()}
    ], lr=1e-3)

#     pretrained_paths = {
#     'nerf_model': r'D:\gzb\NeRF_CLIP\saved_one_img_models\try_one_img_nerf_model_epoch_46100.pth',
#     'shape_mapper': r'D:\gzb\NeRF_CLIP\saved_one_img_models\try_one_img_shape_mapper_epoch_46100.pth',
#     'color_mapper': r'D:\gzb\NeRF_CLIP\saved_one_img_models\try_one_img_color_mapper_epoch_46100.pth',
#     'learnable_camera_pose': r'D:\gzb\NeRF_CLIP\saved_one_img_models\try_one_img_learnable_camera_pose_epoch_46100.pth'
# }
    pretrained_paths = {
    'nerf_model': './day47/multi_view_img_nerf_model_epoch_42000.pth',
    'shape_mapper': './day47/multi_view_img_shape_mapper_epoch_42000.pth',
    'color_mapper':'./day47/multi_view_img_color_mapper_epoch_42000.pth',
    'learnable_camera_pose':'./day47/multi_view_img_learnable_camera_pose_epoch_42000.pth'
}

    models = {
    'nerf_model': nerf_model,
    'shape_mapper': shape_mapper,
    'color_mapper': color_mapper,
    'learnable_camera_pose': learnable_camera_pose
}
# 加载预训练状态
    load_pretrained_models(pretrained_paths, models)
    #生成一张完整的图像
    #将整张图像的所有像素点都选一遍
    K = np.array([
        [focal, 0, 0.5*W],
        [0, focal, 0.5*H],
        [0, 0, 1]
    ])
    new_c2w_matrices = generate_new_views(learnable_camera_pose,15,device)
    #     # print(" img size  ={} * {} ".format(H ,W))
    # text = captions[1]
    # print(text)
    # position, orientation = learnable_camera_pose()
    # rotation_matrix = R.from_quat(orientation.cpu().detach().numpy()).as_matrix()
    # rotation_matrix = torch.tensor(rotation_matrix, dtype=torch.float32).to(device)
    # c2w = torch.eye(4, device=device)
    # c2w[:3, :3] = rotation_matrix
    # c2w[:3, 3] = position
    nums = 0
    for c2w in new_c2w_matrices:
        nums +=1
        N_rays0 = 1024
        t_vals = torch.linspace(0., 1., steps=N_samples)
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))
        rays_o, rays_d = get_rays(H, W, K, c2w) #[H, W, 3]
        # print("ray o ",rays_o.size())
        rays_o = rays_o.reshape(-1, 3) #[H * W , 3 ]
        rays_d = rays_d.reshape(-1, 3) #[H * W , 3 ]
        rgb = torch.zeros((H,W,3),dtype=torch.float32)
        print("rgb size = ",rgb.size())
        z_vals0 = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))
        z_vals0 = z_vals0.expand([N_rays0, N_samples])
        
        shape_offset, color_offset, text_embedding = get_code(text, clip_model, shape_mapper, color_mapper)
        # shape_code0 = shape_code0.repeat(N_rays0, 1)  # [N_rays, shape_dim]
        # color_code0 = color_code0.repeat(N_rays0, 1)  # [N_rays, color_dim]
        # shape_code0 = shape_code0[:, None, :].expand(-1, N_samples, -1).reshape(-1, shape_code0.shape[-1])  # [N_rays * N_samples, shape_dim]
        # color_code0 = color_code0[:, None, :].expand(-1, N_samples, -1).reshape(-1, color_code0.shape[-1])  # [N_rays * N_samples, shape_dim]

        for j in range(int(W/2)):
            selected_rays_o, seleted_rays_d = rays_o[j * N_rays0: (j + 1) * N_rays0], rays_d[j * N_rays0: (j + 1) * N_rays0]
            viewdirs = seleted_rays_d
            viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
            viewdirs = torch.reshape(viewdirs, [-1, 3]).float()
            pts = selected_rays_o[..., None, :] + seleted_rays_d[..., None, :] * z_vals0[..., :,
                                                                                    None]  # None 相当于添加1维度， 如rays_o[N_rays，3] -> [N_rays,1,3]
            # #                          [N_rays ,1,3] *[N_rays, N_samples, 1] = [N_rays, N_Samples,3]
            embedded = get_embeddings(pts, viewdirs, embed_fn, embeddirs_fn,shape_offset, color_offset)
            # embedded = torch.cat([embedded, shape_offset, color_offset], -1)
            raw = nerf_model(embedded)
            outputs = torch.reshape(raw, list(pts.shape[:-1]) + [raw.shape[-1]])
            rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(outputs, z_vals0, seleted_rays_d)
            rgb_map0 = rgb_map.detach()
            # print("rgb_map size ",rgb_map0.size())
            rgb[j*2] = rgb_map0[:512]
            rgb[j*2+1] = rgb_map0[512:]
            print(" 释放显存后分配显存 为 {} GB".format(torch.cuda.memory_allocated('cuda:0') / 1e9))
            # for row in range(0,2):
            #     rgb[j+row] = rgb_map0[row*H:(row+1)*H]
        print(rgb.size())
        import matplotlib.pyplot as plt
        rgb = (rgb*255.0).to(torch.uint8).cpu()
        plt.imshow(rgb)
        plt.imsave(f'./data/rendered_imgs/47hot_dogs_multi_view_{nums}.jpg',rgb.numpy())
        # plt.show()
    # image = generate_image(rgb, H, W)
    # image.show()    


def generate_one_img(H,W,text):
     # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_fn, input_ch = get_embedder(10, 0)
    # Set render parameters
    N_rays = 224
    focal = 1     # Focal length
    near = 1
    far = 2
    N_samples = 128
    embeddirs_fn, input_ch_views = get_embedder(4, 0)
    # Load or initialize Nerf model
    nerf_model = NeRF(input_ch= input_ch, input_ch_views= input_ch_views,use_viewdirs=True).to(device)

    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    # shape_mapper , color_mapper = init_mappers(512,512)
    shape_mapper , color_mapper = init_mappers(input_ch, input_ch_views)
    learnable_camera_pose = LearnableCameraPose().to(device)
    optimizer = torch.optim.Adam([
        {'params': nerf_model.parameters()},
        {'params': shape_mapper.parameters()},
        {'params': color_mapper.parameters()},
        {'params':learnable_camera_pose.parameters()}
    ], lr=1e-3)

#     pretrained_paths = {
#     'nerf_model': r'D:\gzb\NeRF_CLIP\saved_one_img_models\try_one_img_nerf_model_epoch_330.pth',
#     'shape_mapper': r'D:\gzb\NeRF_CLIP\saved_one_img_models\try_one_img_shape_mapper_epoch_330.pth',
#     'color_mapper': r'D:\gzb\NeRF_CLIP\saved_one_img_models\try_one_img_color_mapper_epoch_330.pth',
#     'learnable_camera_pose': r'D:\gzb\NeRF_CLIP\saved_one_img_models\try_one_img_learnable_camera_pose_epoch_330.pth'
# }
    pretrained_paths = {
    'nerf_model': './day47/multi_view_img_nerf_model_epoch_42000.pth',
    'shape_mapper': './day47/multi_view_img_shape_mapper_epoch_42000.pth',
    'color_mapper':'./day47/multi_view_img_color_mapper_epoch_42000.pth',
    'learnable_camera_pose':'./day47/multi_view_img_learnable_camera_pose_epoch_42000.pth'
}
    
    models = {
    'nerf_model': nerf_model,
    'shape_mapper': shape_mapper,
    'color_mapper': color_mapper,
    'learnable_camera_pose': learnable_camera_pose
}
# 加载预训练状态
    load_pretrained_models(pretrained_paths, models)
    #生成一张完整的图像
    #将整张图像的所有像素点都选一遍
    K = np.array([
        [focal, 0, 0.5*W],
        [0, focal, 0.5*H],
        [0, 0, 1]
    ])
    #     # print(" img size  ={} * {} ".format(H ,W))
    # text = captions[1]
    # print(text)
    position, orientation = learnable_camera_pose()
    rotation_matrix = R.from_quat(orientation.cpu().detach().numpy()).as_matrix()
    rotation_matrix = torch.tensor(rotation_matrix, dtype=torch.float32).to(device)
    c2w = torch.eye(4, device=device)
    c2w[:3, :3] = rotation_matrix
    c2w[:3, 3] = position

    N_rays0 = 512
    t_vals = torch.linspace(0., 1., steps=N_samples)
    rays_o, rays_d = get_rays(H, W, K, c2w) #[H, W, 3]
    # print("ray o ",rays_o.size())
    rays_o = rays_o.reshape(-1, 3) #[H * W , 3 ]
    rays_d = rays_d.reshape(-1, 3) #[H * W , 3 ]
    rgb = torch.zeros((H,W,3),dtype=torch.float32)
    print("rgb size = ",rgb.size())
    z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))
    z_vals = z_vals.expand([N_rays0, N_samples])
    
    shape_offset, color_offset, text_embedding = get_code(text, clip_model, shape_mapper, color_mapper)
    # shape_code0 = shape_code0.repeat(N_rays0, 1)  # [N_rays, shape_dim]
    # color_code0 = color_code0.repeat(N_rays0, 1)  # [N_rays, color_dim]
    # shape_code0 = shape_code0[:, None, :].expand(-1, N_samples, -1).reshape(-1, shape_code0.shape[-1])  # [N_rays * N_samples, shape_dim]
    # color_code0 = color_code0[:, None, :].expand(-1, N_samples, -1).reshape(-1, color_code0.shape[-1])  # [N_rays * N_samples, shape_dim]

    for j in range(int(W)):
        selected_rays_o, seleted_rays_d = rays_o[j * N_rays0: (j + 1) * N_rays0], rays_d[j * N_rays0: (j + 1) * N_rays0]
        viewdirs = seleted_rays_d
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()
        pts = selected_rays_o[..., None, :] + seleted_rays_d[..., None, :] * z_vals[..., :,
                                                                                None]  # None 相当于添加1维度， 如rays_o[N_rays，3] -> [N_rays,1,3]
        # #                          [N_rays ,1,3] *[N_rays, N_samples, 1] = [N_rays, N_Samples,3]
        embedded = get_embeddings(pts, viewdirs, embed_fn, embeddirs_fn,shape_offset,color_offset)
        # embedded = torch.cat([embedded, shape_code0, color_code0], -1)
        raw = nerf_model(embedded)
        outputs = torch.reshape(raw, list(pts.shape[:-1]) + [raw.shape[-1]])
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(outputs, z_vals, seleted_rays_d)
        rgb_map0 = rgb_map.detach()
        rgb[j] = rgb_map0

        # print("rgb_map size ",rgb_map0.size())
        # rgb[j*2] = rgb_map0[:512]
        # rgb[j*2+1] = rgb_map0[512:]
        # for row in range(0,2):
        #     rgb[j+row] = rgb_map0[row*H:(row+1)*H]
    print(rgb.size())
    import matplotlib.pyplot as plt
    gen_img = rgb.permute(2,0,1).unsqueeze(0)
    gen_img = F.upsample_bilinear(gen_img,(224,224))
    feature = clip_model.encode_image(gen_img)
    similarity = F.cosine_similarity(feature,text_embedding)
    print("similarity = ",similarity.item())
    rgb = (rgb*255.0).to(torch.uint8).cpu()
    plt.imshow(rgb)
    plt.show()
    # image = generate_image(rgb, H, W)
    # image.show()    




def random_select_rays_train():
    random.seed(0)
    source_path = './data/multi_view_imgs/'
    image_ids, captions = load_datas('./data/multi_view_imgs/captions.txt')
     # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_fn, input_ch = get_embedder(10, 0)
    # Set render parameters
    N_rays = 4096
    sample_scale = 64
    focal = 1     # Focal length
    near = 1
    far = 2
    N_samples = 32
    embeddirs_fn, input_ch_views = get_embedder(4, 0)
    # Load or initialize Nerf model
    nerf_model = NeRF(input_ch= input_ch, input_ch_views= input_ch_views,use_viewdirs=True).to(device)

    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    # shape_mapper , color_mapper = init_mappers(512,512)
    shape_mapper , color_mapper = init_mappers(input_ch, input_ch_views)
    learnable_camera_pose = LearnableCameraPose().to(device)
    optimizer = torch.optim.Adam([
        {'params': nerf_model.parameters()},
        {'params': shape_mapper.parameters()},
        {'params': color_mapper.parameters()},
        {'params':learnable_camera_pose.parameters()}
    ], lr=1e-5)

    pretrained_paths = {
    'nerf_model': r'D:\gzb\NeRF_CLIP\saved_one_img_models\try_one_img_nerf_model_epoch_000.pth',
    'shape_mapper': r'D:\gzb\NeRF_CLIP\saved_one_img_models\try_one_img_shape_mapper_epoch_000.pth',
    'color_mapper': r'D:\gzb\NeRF_CLIP\saved_one_img_models\try_one_img_color_mapper_epoch_000.pth',
    'learnable_camera_pose': r'D:\gzb\NeRF_CLIP\saved_one_img_models\try_one_img_learnable_camera_pose_epoch_20000.pth'
}
    
    models = {
    'nerf_model': nerf_model,
    'shape_mapper': shape_mapper,
    'color_mapper': color_mapper,
    'learnable_camera_pose': learnable_camera_pose
}
# 加载预训练状态
    load_pretrained_models(pretrained_paths, models)
    #相机的内参矩阵
   
    num_epochs = 2000
    total_epoch = 0 #之前训练的总轮数
    total_num_of_epoch = 100
    sum_of_num=0
    for iter in range(total_num_of_epoch):
        for id in range(len(image_ids)):
            img_dir = os.path.join(source_path,image_ids[id])
            img = Image.open(img_dir)
            img_tensor = torch.tensor(np.array(img), dtype=torch.float, device=device)
            # img_tensor = img_tensor.reshape(-1,3)
            # print(img_tensor.size())
            H, W = img.height, img.width
            # print("img size = ",H,W)
            K = np.array([ 
                [focal, 0, 0.5*W],
                [0, focal, 0.5*H],
                [0, 0, 1]
            ])
            text = captions[id]
            # print(f"循环之前分配的显存: {torch.cuda.memory_allocated('cuda:0') / 1e9} GB")
            for epoch in range(1,num_epochs+1):
                sum_of_num+=1
                position, orientation = learnable_camera_pose()
                rotation_matrix = R.from_quat(orientation.cpu().detach().numpy()).as_matrix()
                rotation_matrix = torch.tensor(rotation_matrix, dtype=torch.float32).to(device)
                c2w = torch.eye(4, device=device)
                c2w[:3, :3] = rotation_matrix
                c2w[:3, 3] = position
                del position,rotation_matrix
                torch.cuda.empty_cache()
                t_vals = torch.linspace(0., 1., steps=N_samples)
                z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))
                z_vals = z_vals.expand([N_rays, N_samples])
                rays_o, rays_d = get_rays(H, W, K, c2w) #[H, W, 3]

                img_tensor = img_tensor.reshape(-1,3)
                rays_o = rays_o.reshape(-1, 3) #[H * W , 3 ]
                rays_d = rays_d.reshape(-1, 3) #[H * W , 3 ]
                shape_offset, color_offset, text_embedding = get_code(text, clip_model, shape_mapper, color_mapper)
                # print(f"循环开始分配的显存: {torch.cuda.memory_allocated('cuda:0') / 1e9} GB")
                # start_time = time.time()
                selected_indices = torch.randint(0, H * W, (N_rays,))
                target_s = img_tensor[selected_indices]
                selected_rays_o = rays_o[selected_indices]
                selected_rays_d = rays_d[selected_indices]
                viewdirs = selected_rays_d
                viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
                viewdirs = torch.reshape(viewdirs, [-1, 3]).float()
                pts = selected_rays_o[..., None, :] + selected_rays_d[..., None, :] * z_vals[..., :,
                                                                                        None]  # None 相当于添加1维度， 如rays_o[N_rays，3] -> [N_rays,1,3]
                # #                          [N_rays ,1,3] *[N_rays, N_samples, 1] = [N_rays, N_Samples,3]
                

                pts = selected_rays_o[...,None,:] + selected_rays_d[...,None,:]*z_vals[...,:,None]
                embedded = get_embeddings(pts, viewdirs, embed_fn, embeddirs_fn, shape_offset, color_offset)
                
                # embedded = torch.cat([embedded, shape_code, color_code], -1)
                # print("embeed size = {}, shape code size = {}".format(embedded.size(), shape_code.size()))
                raw = nerf_model(embedded)
                outputs = torch.reshape(raw, list(pts.shape[:-1]) + [raw.shape[-1]])
                # rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(outputs, z_vals, selected_rays_d)
                rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(outputs, z_vals, selected_rays_d )
                gen_img = rgb_map * 255.0
                optimizer.zero_grad()
                mse_loss = F.mse_loss(gen_img,target_s)
                # loss = mse_loss
                # print("rgb_map size ",rgb_map.size())
                rendered_img = rgb_map.view(sample_scale,sample_scale,-1).permute(2,0,1).unsqueeze(0)
                # print("rendered img size ", rendered_img.size())
                scale_img = F.upsample_bilinear(rendered_img,(224,224))
                # print("scaled img size ",scale_img.size())
                # print(text_embedding.size())
                img_feature = clip_model.encode_image(scale_img)
                similarity =  torch.cosine_similarity(img_feature,text_embedding)
                clip_loss = (1 -similarity )
                print("img {} , epoch {}, similarity = {}, mse loss = {}".format(id+1, sum_of_num,similarity,mse_loss))
                
                loss = clip_loss  + mse_loss 
                # print("img {}  epoch {}  total_loss = {} ".format(id,sum_of_num,loss))
                loss.backward()
                optimizer.step()
                # print(" 释放显存后分配显存 为 {} GB".format(torch.cuda.memory_allocated('cuda:0') / 1e9))
                # endtime =time.time()
                # print("epoch {} use time = {} s".format(sum_of_num,endtime - start_time))
                
                if epoch%1000==0:
                   
                    rays_o1, rays_d1 = get_rays(H, W, K, c2w) #[H, W, 3]
                    rays_o1 = rays_o1.reshape(-1, 3) #[H * W , 3 ]
                    rays_d1 = rays_d1.reshape(-1, 3) #[H * W , 3 ]
                    rgb = torch.zeros((H,W,3),dtype=torch.float32)
                    with torch.no_grad():
                        for row in range(H):
                            t_vals0 = torch.linspace(0., 1., steps=N_samples)
                            z_vals0 = 1./(1./near * (1.-t_vals0) + 1./far * (t_vals0))
                            z_vals0 = z_vals0.expand([H, N_samples])
                            selected_rays_o, seleted_rays_d = rays_o1[row * H: (row + 1) * H], rays_d1[row * H: (row+ 1) * H]
                            viewdirs0 = seleted_rays_d
                            viewdirs0 = viewdirs0 / torch.norm(viewdirs0, dim=-1, keepdim=True)
                            viewdirs0 = torch.reshape(viewdirs0, [-1, 3]).float()
                            pts0 = selected_rays_o[..., None, :] + seleted_rays_d[..., None, :] * z_vals0[..., :,
                                                                                                    None]  # None 相当于添加1维度， 如rays_o[N_rays，3] -> [N_rays,1,3]
                            # #                          [N_rays ,1,3] *[N_rays, N_samples, 1] = [N_rays, N_Samples,3]
                            shape_code0, color_code0, text_embedding0 = get_code2(text, clip_model, shape_mapper, color_mapper,H,N_samples)
                            embedded0 = get_embeddings(pts0.detach(), viewdirs0, embed_fn, embeddirs_fn)
                            embedded0 = torch.cat([embedded0, shape_code0, color_code0], -1)
                            raw0 = nerf_model(embedded0)
                            outputs0 = torch.reshape(raw0, list(pts0.shape[:-1]) + [raw0.shape[-1]])
                            rgb_map0, disp_map, acc_map, weights, depth_map = raw2outputs(outputs0, z_vals0, seleted_rays_d)
                            rgb_map1 = rgb_map0.detach()
                            # print("rgb_map size ",rgb_map0.size())
                            rgb[row] = rgb_map1
                        rgb = (rgb*255.0).to(torch.uint8).cpu()
                        plt.imsave(f'./data/new_rendered_imgs/batch_{iter}_img_{id+1}_view_{epoch//1000}.jpg',rgb.numpy())
        save_one_img_models(sum_of_num,models,optimizer)              
                

if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # make_train(1)
    # try_train()
    one_img_train2()
    # generate_one_img(512,512,"There are two golden hot dogs on a plate, and there is also yellow cream on the hot dogs.")
    # random_select_rays_train()
    # generate_multi_perspective_imgs(512,512,"There are two golden hot dogs on a plate, and there is also yellow cream on the hot dogs.")

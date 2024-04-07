import torch
import torch.nn.functional as F
import numpy as np


def get_rays_np(H, W, K, c2w):
    i , j = np.meshgrid(np.arange(W, dtype=np.float32),np.arange(H, dtype=np.float32),indexing='xy')
    # 2D点到3D点的映射计算，[x,y,z]=[(u-cx)/fx,-(-v-cy)/fx,-1]
    # 在y和z轴均取相反数，因为nerf使用的坐标系x轴向右，y轴向上，z轴向外；
    # dirs的大小为(378, 504, 3)
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    # 将ray方向从相机坐标系转到世界坐标系，矩阵不变
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    # 相机原点在世界坐标系的坐标，同一个相机所有ray的起点；
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))  # [1024,3]
    return rays_o, rays_d

def get_rays(H, W, K, c2w):
    """
    生成射线。
    H, W: 图像的高度和宽度。
    K: 相机内参矩阵。
    c2w: 相机的世界坐标系下的位置和朝向（相机到世界的变换矩阵）。
    """
    i, j = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H))  # 像素网格
    i = i.t()
    j = j.t()

    dirs = torch.stack([(i - K[0, 2]) / K[0, 0], -(j - K[1, 2]) / K[1, 1], -torch.ones_like(i)], -1)  # 每个像素对应的射线方向
    # 将射线方向从相机坐标系转换到世界坐标系
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], -1)
    rays_o = c2w[:3, -1].expand(rays_d.shape)  # 所有射线的起点都是相机的位置
    return rays_o, rays_d




def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
        raw_noise_std: Standard deviation of noise added to raw predictions.
        white_bkgd: Whether to assume a white background.
        pytest: Whether running pytest.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape).to(dists.device)], -1)

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
    print("raw size  = ",raw.size())
    rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    print(raw.size())
    # print(noise.size())
    print(dists.size())
    alpha = raw2alpha(raw[..., 3] + noise, dists.reshape(raw.shape[0], -1))
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    return {
        'rgb_map': rgb_map,
        'disp_map': disp_map,
        'acc_map': acc_map,
        'weights': weights,
        'depth_map': depth_map
    }




def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
    """Volumetric rendering.体素渲染
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
        用来view_ray采样的所有必需数据：ray原点、ray方向、最大最小距离、方向单位向量；
      network_fn: function. Model for predicting RGB and density at each point
        in space.
        nerf网络，用来预测空间中每个点的RGB和不透明度的函数
      network_query_fn: function used for passing queries to network_fn.
        将查询传递给network_fn的函数
      N_samples: int. Number of different times to sample along each ray.coarse采样点数
      retraw: bool. If True, include model's raw, unprocessed predictions.是否压缩数据
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.在深度图上面逆向线性采样；
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.扰动
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.fine增加的精细采样点数；
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    # 将数据提取出来
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples)  # 0-1线性采样N_samples个值
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    # 加入扰动
    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    # 每个采样点的3D坐标，o+td
    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]，torch.Size([1024, 64, 3])


#     raw = run_network(pts)
    raw = network_query_fn(pts, viewdirs, network_fn)  # 送进网络进行预测，前向传播；
    # 体素渲染！将光束颜色合成图像上的点
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    # fine网络情况，再次计算上述步骤，只是采样点不同；
    if N_importance > 0:

        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
#         raw = run_network(pts, fn=run_fn)
        raw = network_query_fn(pts, viewdirs, run_fn)

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret




#批量处理射线进行渲染
def batchify_rays(rays_flat,shape_code, color_code, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
       rays_flat ：扁平化的射线数组，包括射线的原点、方向、可能的近平面和远平面
       chunk 定义了每一批次的射线数量
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret



def render(H, W, K,shape_code, color_code, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
    """
    H 图像的高度
    W 图像的宽度
    shape_code 用于控制nerf中体密度的生成
    color_code 用于控制nerf中 color 的生成
    focal 针孔相机焦距
    chunk 同时处理的最大光线数
    rays [2, batch_size, 3]光线， 其中2代表每一个batch的原点和方向
    c2w [3,4] 相机世界到真实世界的旋转矩阵
    ndc NDC coordinates,如果是true 表示射线的原点和方向使用的是归一化设备坐标
    near , far  射线最近和最远的距离
    use_viewdirs true表示使用空间点的观察方向作为输入
    c2w_staticcam 如果非None,这个变换矩阵用于相机,而c2w参数用于视觉方向。
    return :
    rgb_map 预测的RGB图
    disp_map 视差图
    acc_map 不透明度
    """
    #如果提供了c2w矩阵，则根据图像的尺寸和相机参数计算整个图像的每一个像素对应的射线的原点和方向
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:#否则直接使用这些射线
        # use provided ray batch
        rays_o, rays_d = rays

    #如果要使用viewdir，则把计算好的射线的方向赋值给viewdirs
    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # 创建射线处理批次， 将射线的原点、方向、near、far 和视角方向合并为一个批处理组，进行批量渲染
    rays_o = torch.reshape(rays_o, [-1,3]).float()  # torch.Size([1024, 3])
    rays_d = torch.reshape(rays_d, [-1,3]).float()  # torch.Size([1024, 3])

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)  # torch.Size([1024, 8])
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)  # torch.Size([1024, 11])

    # Render and reshape
    all_ret = batchify_rays(rays,shape_code, color_code, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]

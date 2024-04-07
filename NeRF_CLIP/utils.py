import numpy as np
from PIL import Image

def generate_image(rgb_map, H, W):

    # 将RGB颜色值从[0, 1]范围映射到[0, 255]范围，并转换为整数类型
    rgb_map = np.clip(rgb_map * 255, 0, 255).astype(np.uint8)
    
    # 创建一个空白图像，大小为(H, W)，默认黑色背景
    image = np.zeros((H, W, 3), dtype=np.uint8)

    # 将RGB颜色值填充到图像中的每个像素上
    for i in range(rgb_map.shape[0]):
        image[i, :, :] = rgb_map[i, :]

    # 使用PIL库将numpy数组转换为图像
    image = Image.fromarray(image)
    
    return image


def random_camera_pose():
    # 在球面上均匀分布地随机选择相机的朝向
    theta = torch.rand(1) * 2 * np.pi  # 绕y轴的旋转角度
    phi = torch.rand(1) * np.pi - np.pi / 2  # 绕x轴的旋转角度
    # 计算相机的方向向量
    direction = torch.tensor([torch.cos(theta) * torch.cos(phi),
                              torch.sin(phi),
                              torch.sin(theta) * torch.cos(phi)])
    # 随机选择相机的位置
    position = torch.rand(3) * 20 - 10  # 在范围(-10, 10)内随机选择位置
    # 生成相机外参矩阵
    c2w = torch.eye(4, dtype=torch.float32)
    c2w[:3, :3] = torch.eye(3)  # 单位旋转矩阵
    c2w[:3, 3] = position  # 设置相机位置
    return c2w


def random_camera_pose2():
    # 在球面上均匀分布地随机选择相机的朝向
    theta = np.random.uniform(0, 2 * np.pi)  # 绕y轴的旋转角度
    phi = np.random.uniform(-np.pi / 2, np.pi / 2)  # 绕x轴的旋转角度
    # 计算相机的方向向量
    direction = np.array([np.cos(theta) * np.cos(phi),
                          np.sin(phi),
                          np.sin(theta) * np.cos(phi)])
    # 随机选择相机的位置
    position = np.random.uniform(-10, 10, size=3)  # 在范围(-10, 10)内随机选择位置
    # 生成相机外参矩阵
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = np.eye(3)  # 单位旋转矩阵
    c2w[:3, 3] = position  # 设置相机位置
    return c2w


def sample_rays(rays_o, rays_d, N_samples):
    """在每个射线上进行均匀采样"""
    # 计算每个射线上的采样点的深度值
    t_vals = np.linspace(0., 1., num=N_samples)  # 在标准化深度范围内均匀采样
    t_vals = np.expand_dims(t_vals, axis=(0, 1))  # 扩展 t_vals 的维度以匹配 (H, W, N_samples)
    z_vals = rays_o[..., 2:3] + t_vals * rays_d[..., 2:3]  # 根据射线方向计算采样点的深度

    # 根据深度值计算采样点的位置jh   
    sampled_rays_o = np.expand_dims(rays_o, axis=1)  # 在第二个维度上扩展
    sampled_rays_o = np.tile(sampled_rays_o, (1, N_samples, 1))  # 扩展起点的尺寸以匹配深度值
    sampled_rays_d = np.expand_dims(rays_d, axis=1)  # 在第二个维度上扩展
    sampled_rays_d = np.tile(sampled_rays_d, (1, N_samples, 1))  # 扩展方向的尺寸以匹配深度值
    sampled_points = sampled_rays_o + z_vals * sampled_rays_d  # 计算采样点的位置

    return sampled_points


def inference2():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_fn, input_ch = get_embedder(10, 0)
    # Set render parameters
    H, W = 256, 256  # Image dimensions
    focal = 1     # Focal length
    N_rays = 256
    N_samples = 32
    embeddirs_fn, input_ch_views = get_embedder(4, 0)
    # Load or initialize Nerf model
    nerf_model = NeRF(input_ch= input_ch, input_ch_views= input_ch_views,use_viewdirs=True).to(device)

    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    text = ["a blue car"]

    shape_code, color_code,text_embedding = get_code(text,clip_model)

    shape_code = shape_code.repeat(N_rays, 1)  # [N_rays, shape_dim]
    color_code = color_code.repeat(N_rays, 1) # [N_rays, color_dim]
    shape_code = shape_code[:, None, :].expand(-1, N_samples, -1).reshape(-1, shape_code.shape[-1]) #[N_rays * N_samples, shape_dim]
    color_code = color_code[:, None, :].expand(-1, N_samples, -1).reshape(-1, color_code.shape[-1])#[N_rays * N_samples, shape_dim]
    # print("shape code, color code  size = ",shape_code.size())
    # Define camera parameters
    c2w = random_camera_pose()
    
    K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])
    near = 1
    far = 6
    t_vals = torch.linspace(0., 1., steps=N_samples)
    z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))
    z_vals = z_vals.expand([N_rays, N_samples])
    rays_o, rays_d = get_rays(H, W, K, c2w) #[H, W, 3]
    # print("ray o ",rays_o.size())
    rays_o = rays_o.reshape(-1, 3) #[H * W , 3 ]
    rays_d = rays_d.reshape(-1, 3) #[H * W , 3 ]
    # print("ray o ",rays_o.size())
    first_selected = False
    rgb = np.empty([H,W,3],dtype=float)
    # print(rgb.shape)


    for i in range(int(H * W / N_rays)):
        selected_rays_o, seleted_rays_d = rays_o[i * N_rays: (i + 1) *N_rays], rays_d[i * N_rays: (i + 1) *N_rays]
        viewdirs = seleted_rays_d
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()
        pts = selected_rays_o[...,None,:] + seleted_rays_d[...,None,:] * z_vals[...,:,None] #None 相当于添加1维度， 如rays_o[N_rays，3] -> [N_rays,1,3]
    # #                          [N_rays ,1,3] *[N_rays, N_samples, 1] = [N_rays, N_Samples,3]             

        # inputs_flat = torch.reshape(pts, [-1, pts.shape[-1]]) #[N_rays * N_samples, 3]
        # embedded = embed_fn(inputs_flat)#[N_rays * N_samples, 63] 其中63  =  2 * 10 * 3 +3

        # input_dirs = viewdirs[:,None].expand(pts.shape) #[N_rays, N_samples, 3]
        # input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]]) #[N_rays * N_samples, 3]
        # embedded_dirs = embeddirs_fn(input_dirs_flat) #[N_rays * N_samples, 27] 27 = 3 * 4*2 + 3
        # embedded = torch.cat([embedded, embedded_dirs], -1) #[N_rays * N_samples, 90] 90 = 63 + 27
        embedded = get_embeddings(pts, viewdirs, embed_fn, embeddirs_fn)
        
        # if first_selected == False:
        #     # # shape_code和color_code[ shape/color_dim , 1]
        #     shape_code = shape_code.repeat(pts.shape[0], 1)  # [N_rays, shape_dim]
        #     color_code = color_code.repeat(pts.shape[0], 1) # [N_rays, color_dim]
        #     first_selected = True
        #     shape_code = shape_code[:, None, :].expand(-1, pts.shape[1], -1).reshape(-1, shape_code.shape[-1]) #[N_rays * N_samples, shape_dim]
        #     color_code = color_code[:, None, :].expand(-1, pts.shape[1], -1).reshape(-1, color_code.shape[-1]) #[N_rays * N_samples, color_dim]
        
        
        
        embedded = torch.cat([embedded, shape_code, color_code], -1)
        # print("final embedded size = " , embedded.size())
        raw = nerf_model(embedded)
        # print("raw size = ",raw.size())
        outputs = torch.reshape(raw, list(pts.shape[:-1]) + [raw.shape[-1]])
        # print("outputs size = ",outputs.size())
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(outputs,z_vals,seleted_rays_d)
        rgb_map = rgb_map.detach().cpu().numpy()
        rgb[i] = rgb_map
        # print("cur rbg_ map {}".format(len(rgb)))
    print(rgb.shape)
    image = generate_image(rgb, H,W)
    # image.show()
    preprocessed_image = preprocess(image).unsqueeze(0).to('cuda')
    with torch.no_grad():
        image_embedding = clip_model.encode_image(preprocessed_image)
    similarity = torch.nn.functional.cosine_similarity(text_embedding, image_embedding)
    print(similarity)
    image.show()




def inference():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_fn, input_ch = get_embedder(10, 0)
    # Set render parameters
    H, W = 256, 256  # Image dimensions
    focal = 1     # Focal length
    chunk = 1024      # Chunk size for rendering
    embeddirs_fn, input_ch_views = get_embedder(4, 0)
    # Load or initialize Nerf model
    nerf_model = NeRF(input_ch= input_ch, input_ch_views= input_ch_views,use_viewdirs=True).to(device)
    # nerf_model.input_ch = input_ch
    # nerf_model.input_ch_views = input_ch_views
    text = ["a blue car"]
    shape_code, color_code,text_embedding,clip_model = get_code(text)
    print("shape code, color code  size = ",shape_code.size())

    # Define camera parameters
    c2w = random_camera_pose()
    
    K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])
    rays_o, rays_d = get_rays(H, W, K, c2w) #[N_rays, N_samples, 3]
    print("ray o ",rays_o.size())
    num_rays = rays_o.shape[0]
    N_rays = 256
    N_samples = 32
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3) #[N_rays * N_samples, 3]
    # 随机选择 N_rays 个索引
    selected_indices = np.random.choice(num_rays, size=N_rays, replace=False)

    # print("sele ",selected_indices.shape)
    rays_o, rays_d = rays_o[selected_indices], rays_d[selected_indices]
    # print("ray o ",rays_o.size())
    rays_o,rays_d = torch.tensor(rays_o),torch.tensor(rays_d)

    # print("Device type:", type(rays_d))
    viewdirs = rays_d
    # viewdirs= torch.tensor(viewdirs, dtype=torch.float32)
    near = 1
    far = 6
 
    viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
    viewdirs = torch.reshape(viewdirs, [-1,3]).float()
    # pts = sample_rays(rays_o,rays_d,32)

    t_vals = torch.linspace(0., 1., steps=N_samples)
    z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))
    z_vals = z_vals.expand([N_rays, N_samples])
    # print("z val ",z_vals.size())
    # print('rays_o[...,None,:]',rays_o[...,None,:].size())
    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] #None 相当于添加1维度， 如rays_o[N_rays，3] -> [N_rays,1,3]
    #                          [N_rays ,1,3] *[N_rays, N_samples, 1] = [N_rays, N_Samples,3]             
    print("points size ", pts.size())


    inputs_flat = torch.reshape(pts, [-1, pts.shape[-1]]) #[N_rays * N_samples, 3]
    embedded = embed_fn(inputs_flat)#[N_rays * N_samples, 63] 其中63  =  2 * 10 * 3 +3

    input_dirs = viewdirs[:,None].expand(pts.shape) #[N_rays, N_samples, 3]
    input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]]) #[N_rays * N_samples, 3]
    embedded_dirs = embeddirs_fn(input_dirs_flat) #[N_rays * N_samples, 27] 27 = 3 * 4*2 + 3
    embedded = torch.cat([embedded, embedded_dirs], -1) #[N_rays * N_samples, 90] 90 = 63 + 27
    print("embedded size = ",embedded.size())
    # 假设shape_code和color_code是每个光线的属性，需要扩展为每个采样点
    # shape_code和color_code[ shape/color_dim , 1]
    shape_code = shape_code.repeat(pts.shape[0], 1)  # [N_rays, shape_dim]
    color_code = color_code.repeat(pts.shape[0], 1) # [N_rays, color_dim]


    shape_code_expanded = shape_code[:, None, :].expand(-1, pts.shape[1], -1).reshape(-1, shape_code.shape[-1]) #[N_rays * N_samples, shape_dim]
    color_code_expanded = color_code[:, None, :].expand(-1, pts.shape[1], -1).reshape(-1, color_code.shape[-1]) #[N_rays * N_samples, color_dim]


    embedded = torch.cat([embedded, shape_code_expanded, color_code_expanded], -1)
    print("final embedded size = " , embedded.size())
    raw = nerf_model(embedded)
    print("raw size = ",raw.size())
    outputs = torch.reshape(raw, list(pts.shape[:-1]) + [raw.shape[-1]])
    print("outputs size = ",outputs.size())
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(outputs,z_vals,rays_d)
    rgb_map = rgb_map.detach().cpu().numpy()
    print(rgb_map.shape)
    # image = generate_image(rgb_map, H,W)
    # from torchvision import transforms
    # transform = transforms.ToTensor()
    # image_tensor = transform(image)
    # similarity = clip_model(text_embedding, image_tensor.unsqueeze(0))
    # print(similarity)
    # image.show()





#原始版本  训练函数
# def try_train():
   
#     source_path = './data/new_imgs/'
#     image_ids, captions = load_datas()
#      # Set device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     embed_fn, input_ch = get_embedder(10, 0)
#     # Set render parameters
#     N_rays = 1024
#     focal = 1     # Focal length
#     near = 1
#     far = 6
#     N_samples = 64
#     embeddirs_fn, input_ch_views = get_embedder(4, 0)
#     # Load or initialize Nerf model
#     nerf_model = NeRF(input_ch= input_ch, input_ch_views= input_ch_views,use_viewdirs=True).to(device)

#     clip_model, preprocess = clip.load("ViT-B/32", device='cpu')
#     shape_mapper , color_mapper = init_mappers(512,512)

#     optimizer = torch.optim.Adam([
#         {'params': nerf_model.parameters()},
#         {'params': shape_mapper.parameters()},
#         {'params': color_mapper.parameters()}
#     ], lr=1e-3)



#     num_epochs = 1000
    
#     for i in range(len(image_ids)):
#         img_dir = os.path.join(source_path,image_ids[i])
#         img = Image.open(img_dir)
#         img_tensor = torch.tensor(np.array(img), dtype=torch.float, device=device)
#         img_tensor = img_tensor.reshape(-1,3)
#         H, W = img.height, img.width
#         print(" img size  ={} * {} ".format(H ,W))
#         text = captions[i]

#         # shape_code0 = shape_code
#         # color_code0 = color_code

#         c2w = random_camera_pose()
#         K = np.array([
#             [focal, 0, 0.5*W],
#             [0, focal, 0.5*H],
#             [0, 0, 1]
#         ])
#         t_vals = torch.linspace(0., 1., steps=N_samples)
#         z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))
#         z_vals = z_vals.expand([N_rays, N_samples])
#         rays_o, rays_d = get_rays(H, W, K, c2w) #[H, W, 3]
#         # print("ray o ",rays_o.size())
#         rays_o = rays_o.reshape(-1, 3) #[H * W , 3 ]
#         rays_d = rays_d.reshape(-1, 3) #[H * W , 3 ]
#         total_mse_loss = 0.0

#         for epoch in range(num_epochs):
#             shape_code0, color_code0, text_embedding = get_code(text, clip_model, shape_mapper, color_mapper)
#             shape_code = shape_code0.repeat(N_rays, 1)  # [N_rays, shape_dim]
#             color_code = color_code0.repeat(N_rays, 1)  # [N_rays, color_dim]
#             shape_code = shape_code[:, None, :].expand(-1, N_samples, -1).reshape(-1, shape_code.shape[
#                 -1])  # [N_rays * N_samples, shape_dim]
#             color_code = color_code[:, None, :].expand(-1, N_samples, -1).reshape(-1, color_code.shape[
#                 -1])  # [N_rays * N_samples, shape_dim]
#             # rgb = torch.empty((H,W,3),dtype=torch.float)
#             print("image {} epoch {}".format(i, epoch))
#             selected_indices = torch.randint(0, H * W, (N_rays,))
#             img_block = img_tensor[selected_indices]
#             # print("img_block size = ", img_block.size())
#             selected_rays_o = rays_o[selected_indices]
#             selected_rays_d = rays_d[selected_indices]
#             viewdirs = selected_rays_d
#             viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
#             viewdirs = torch.reshape(viewdirs, [-1, 3]).float()
#             pts = selected_rays_o[..., None, :] + selected_rays_d[..., None, :] * z_vals[..., :,None]  # None 相当于添加1维度， 如rays_o[N_rays，3] -> [N_rays,1,3]

#             embedded = get_embeddings(pts, viewdirs, embed_fn, embeddirs_fn)
#             embedded = torch.cat([embedded, shape_code, color_code], -1)
#             # with torch.autograd.set_detect_anomaly(True):
#             raw = nerf_model(embedded)
#             outputs = torch.reshape(raw, list(pts.shape[:-1]) + [raw.shape[-1]])
#             # print("outputs size = ", outputs.size())
#             rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(outputs, z_vals, selected_rays_d)
#             # print("rgb_map size ", rgb_map.size())
#             # img_block = img_tensor[i * N_rays: (i + 1) * N_rays]
#             # print("img_block size = ",img_block.size())
#             mse_loss = F.mse_loss(rgb_map, img_block)
#             print("mse loss = ", mse_loss)
#             # total_mse_loss += mse_loss
#             # print("total loss = ", total_mse_loss)
#             optimizer.zero_grad()
#             mse_loss.backward(retain_graph=True)
#             optimizer.step()



#         # #生成一张完整的图像
#         # # 将整张图像的所有像素点都选一遍
#         # N_rays0 = H
#         # rgb = np.empty([H, W, 3], dtype=float)
#         # z_vals0 = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))
#         # z_vals0 = z_vals0.expand([N_rays0, N_samples])
#         #
#         # shape_code0 = shape_code0.repeat(N_rays0, 1)  # [N_rays, shape_dim]
#         # color_code0 = color_code0.repeat(N_rays0, 1)  # [N_rays, color_dim]
#         # shape_code0 = shape_code0[:, None, :].expand(-1, N_samples, -1).reshape(-1, shape_code0.shape[
#         #     -1])  # [N_rays * N_samples, shape_dim]
#         # color_code0 = color_code0[:, None, :].expand(-1, N_samples, -1).reshape(-1, color_code0.shape[
#         #     -1])  # [N_rays * N_samples, shape_dim]
#         # for j in range(int(H * W / N_rays0)):
#         #     selected_rays_o, seleted_rays_d = rays_o[j * N_rays0: (j + 1) * N_rays0], rays_d[j * N_rays0: (j + 1) * N_rays0]
#         #     print("selece size ",seleted_rays_d.size())
#         #     print("zval size = ",z_vals0.size())
#         #     viewdirs = seleted_rays_d
#         #     viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
#         #     viewdirs = torch.reshape(viewdirs, [-1, 3]).float()
#         #     pts = selected_rays_o[..., None, :] + seleted_rays_d[..., None, :] * z_vals0[..., :,
#         #                                                                          None]  # None 相当于添加1维度， 如rays_o[N_rays，3] -> [N_rays,1,3]
#         #     # #                          [N_rays ,1,3] *[N_rays, N_samples, 1] = [N_rays, N_Samples,3]
#         #     embedded = get_embeddings(pts, viewdirs, embed_fn, embeddirs_fn)
#         #     print("embe  size = {} code size = {}".format(embedded.size(), shape_code0.size()))
#         #     embedded = torch.cat([embedded, shape_code0, color_code0], -1)
#         #     raw = nerf_model(embedded)
#         #     outputs = torch.reshape(raw, list(pts.shape[:-1]) + [raw.shape[-1]])
#         #     rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(outputs, z_vals0, seleted_rays_d)
#         #     rgb_map0 = rgb_map.detach().cpu().numpy()
#         #     rgb[j] = rgb_map0
#         #     image = generate_image(rgb, H, W)
#         #     image.show()

#             #     total_mse_loss += mse_loss
#             #     print("total loss = ",total_mse_loss)



#             # optimizer.zero_grad()
#             # total_mse_loss.backward()
#             # optimizer.step()


#                 # rgb_map = rgb_map.detach().cpu().numpy()
#                 # rgb[i] = rgb_map
#             # predicted_image = generate_image(rgb, H,W)
#             # # image.show()
#             # preprocessed_image = preprocess(predicted_image).unsqueeze(0).to('cuda')
#             # mse_loss = F.mse_loss(rgb,img_tensor)
#             # print("mse loss = ",mse_loss)
#             # with torch.no_grad():
#             #     image_embedding = clip_model.encode_image(preprocessed_image)
#             # similarity = torch.nn.functional.cosine_similarity(text_embedding, image_embedding)
#             # similarity_loss = 1.0 - similarity
#             # print("img {}  similarity = {}".format(img_dir, similarity))
#             # loss = mse_loss + similarity_loss
#             # total_mse_loss += mse_loss.item()
#             # total_similarity_loss += similarity_loss.item()
        
#             # Backward pass
#             # optimizer.zero_grad()
#             # loss.backward()
#             # optimizer.step()




def generate_new_views(learnable_camera_pose, num_views, device):
    # 获取训练好的相机位置和方向
    trained_position, trained_orientation = learnable_camera_pose()
    trained_rotation_matrix = quaternion_to_rotation_matrix(trained_orientation, device)

    new_c2w_matrices = []
    for i in range(num_views):
        # 生成新的相机方向（例如，围绕 y 轴旋转）
        angle = 2 * np.pi * i / num_views
        rotation_matrix = torch.tensor([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ], dtype=torch.float32, device=device)

        # 结合训练好的相机位置和新的相机方向构建 c2w 矩阵
        c2w = torch.eye(4, device=device)
        c2w[:3, :3] = torch.matmul(rotation_matrix, trained_rotation_matrix)
        c2w[:3, 3] = trained_position
        new_c2w_matrices.append(c2w)

    return new_c2w_matrices

# 使用示例
num_views = 60  # 生成 60 个新视角
new_c2w_matrices = generate_new_views(learnable_camera_pose, num_views, device)

# 使用新的 c2w 矩阵渲染多视角图像...

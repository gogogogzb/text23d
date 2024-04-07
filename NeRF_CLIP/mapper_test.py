from mappers import ShapeMapper,ColorMapper
import torch,clip
from load_data import load_datas
from torch.nn import functional as F
device = 'cuda'
def init_mappers(shape_dim, color_dim):
    # 初始化 Shape Mapper 和 Color Mapper
    shape_mapper = ShapeMapper(input_dim=shape_dim, output_dim=shape_dim).to(device=device)
    color_mapper = ColorMapper(input_dim=color_dim, output_dim=color_dim).to(device=device)
    return shape_mapper, color_mapper

def get_code(text, clip_model,shape_mapper, color_mapper): 
    """
    text [batchsize , text size]
    """
    with torch.no_grad():
        tokenized_text = clip.tokenize(text).to('cpu')
        text_embedding =clip_model.encode_text(tokenized_text).float() #[batchsize , 512]
    # print("text size ",text_embedding.size())
    text_embedding = text_embedding.to(device)
    shape_code = shape_mapper(text_embedding)
    color_code = color_mapper(text_embedding)
    print("shape code size ",shape_code.size())
    return shape_code, color_code,text_embedding #[batchsize, 64]

def train():
    
    source_path = './data/new_imgs/'
    image_ids, captions = load_datas()
    shape_mapper, color_mapper = init_mappers(512,512)
    optimizer = torch.optim.Adam([
        {'params': shape_mapper.parameters()},
        {'params': color_mapper.parameters()}
    ], lr=1e-3)
    clip_model, preprocess = clip.load("ViT-B/32", device='cpu')
    for i in range(len(captions)):
        for epoch in range(10):
            text = captions[i]
            image_id = image_ids[i]
            shape_code, color_code ,text_embedding= get_code(text,clip_model,shape_mapper,color_mapper)
            # print(shape_code.size(), text_embedding.size())
            mseloss = F.mse_loss(shape_code,text_embedding) + F.mse_loss(color_code,text_embedding)
            print(mseloss)
            optimizer.zero_grad()
            mseloss.backward()
            optimizer.step()    
       
        
    
train()
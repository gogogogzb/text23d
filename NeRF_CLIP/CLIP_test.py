import torch
import clip
from PIL import Image
from mappers import ShapeMapper,ColorMapper
def init_mappers(shape_dim, color_dim):
    # 初始化 Shape Mapper 和 Color Mapper
    shape_mapper = ShapeMapper(input_dim=shape_dim, output_dim=shape_dim//4)
    color_mapper = ColorMapper(input_dim=color_dim, output_dim=color_dim//4)

    return shape_mapper, color_mapper

def get_text_embedding(text):
    model, preprocess = clip.load("ViT-B/32", device=device)
    tokenized_text = clip.tokenize(text).to(device)
    return model.encode_text(tokenized_text)

def get_code(text_embedding):
    shape_mapper , color_mapper = init_mappers(text_embedding.shape[1],text_embedding.shape[1])
    shape_code = shape_mapper(text_embedding)
    color_code = color_mapper(text_embedding)
    return shape_code, color_code

device = "cpu"
text =["a China map","a big dog", "a large dog"]
text_features = get_text_embedding(text)
print(text_features.size())
shape,color = get_code(text_features)
print(shape.size(), color.size())


# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(text)
# image = preprocess(Image.open("imgs/1.png")).unsqueeze(0).to(device)
# text = clip.tokenize(["a China map", "an America map", "a World map"]).to(device)
# with torch.no_grad():
#     image_features = model.encode_image(image)
#     text_features = model.encode_text(text)
#     text_code = text_features.reshape(-1)
#     print("text.size()", text_code.size())

#     logits_per_image, logits_per_text = model(image, text)
#     probs = logits_per_image.softmax(dim=-1).cpu().numpy()
# print("Label probs:", probs)  
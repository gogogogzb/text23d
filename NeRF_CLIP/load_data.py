# import pandas as pd

# annotations = pd.read_table(r'G:\BaiduDownload\flickr 30k\results_20130124.token', sep='\t', header=None,
#                             names=['image', 'caption'])

import pandas as pd
import os
import shutil
# # 读取数据
# annotations = pd.read_table(r'G:\BaiduDownload\flickr 30k\results_20130124.token', sep='\t', header=None,
#                             names=['image', 'caption'])

# # 提取图像名称和第一个描述（#0）
# annotations['image_id'] = annotations['image'].apply(lambda x: x.split('#')[0])
# annotations['desc_num'] = annotations['image'].apply(lambda x: x.split('#')[1])
# first_descriptions = annotations[annotations['desc_num'] == '0']

# # 如果你只想要前100个图像的第一个描述
# first_100_descriptions = first_descriptions.head(100)
# # 图像原始路径
# source_img_path = r'G:\BaiduDownload\flickr 30k\flickr30k-images'
# # 目标路径
# target_img_path = r'data/imgs'
# os.makedirs(target_img_path,exist_ok= True)
# # 将这些描述写入txt文件
# with open('first_100_descriptions.txt', 'w', encoding='utf-8') as f:
#     for index, row in first_100_descriptions.iterrows():
#         f.write(f"{row['image_id']}: {row['caption']}\n")
#         source_img = os.path.join(source_img_path, row['image_id'])
#         target_img = os.path.join(target_img_path, row['image_id'])
#         shutil.copy(source_img, target_img)


from PIL import Image

def resize_images(source_img_path, out_put_path,size = (512, 512)):
    source_img = Image.open(source_img_path)
    resized_img = source_img.resize(size,Image.BILINEAR)
    resized_img.save(out_put_path)
#读取数据 返回 图像id 以及对应图像描述
def load_datas(caption_dir):
    images_id= []
    captions = []
    # source_path = './data/imgs/'
    # target_path = './data/new_imgs/'
    with open(caption_dir, 'r',encoding='utf-8') as f:
        for line in f:
            image_id, caption = line.strip().split(':', 1)
           
           
            #改变图像大小 为 512 *512
            # img_dir = os.path.join(source_path +image_id)
            # resized_img_path = os.path.join(target_path +image_id)
            # resize_images(img_dir, resized_img_path)

            
            images_id.append(image_id)
            captions.append(caption)
    # print(images_id[:5])
    # img = Image.open(images_id[1])
    # img.show()
    # print(img.width, img.height)
    return images_id, captions




if __name__=='__main__':
    
    images_id, captions = load_datas()
    for i in range(len(images_id[:10])):
        print (images_id[i],captions[i])
        
    





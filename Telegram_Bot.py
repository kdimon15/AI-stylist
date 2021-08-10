import os

import cv2
import faiss
import numpy as np
import telebot
import torch
from torch.utils.data import DataLoader
import face_recognition

from Model import CustomModel
from utils import ClothesDataset, get_transforms, FaceDataset, device

face_paths, clothes_paths = [], []
for path1 in os.listdir('/content/data'):
    face_paths.append(f'data/{path1}/face.jpg')
    for path2 in os.listdir(f'/content/data/{path1}'):
        if path2 != 'face.jpg':
            clothes_paths.append(f'data/{path1}/{path2}')

face_dataset = FaceDataset(face_paths, transform=get_transforms(data='valid', purpose='faces'))
face_loader = DataLoader(
    face_dataset,
    batch_size=100,
    shuffle=False,
    pin_memory=False,
    drop_last=False
)

clothes_dataset = ClothesDataset(clothes_paths, transform=get_transforms(data='valid', purpose='clothes'))
clothes_loader = DataLoader(
    clothes_dataset,
    batch_size=100,
    shuffle=False,
    pin_memory=False,
    drop_last=False
)

face_model = CustomModel()
face_model.load_state_dict(torch.load('Weights/first_face_model.pth', map_location=device))
face_model.eval()
face_model.to(device)

clothes_model = CustomModel()
clothes_model.load_state_dict(torch.load('Weights/first_clothes_model.pth', map_location=device))
clothes_model.eval()
clothes_model.to(device)

face_transform = get_transforms(data='valid', purpose='faces')
clothes_transform = get_transforms(data='valid', purpose='clothes')

all_clothes_preds = []
for clothes_images in clothes_loader:
    clothes_images = clothes_images.to(device)
    preds = clothes_model(clothes_images)
    preds = preds.cpu().detach().numpy()
    all_clothes_preds.append(preds)
all_clothes_preds = np.concatenate(all_clothes_preds)


d = 1280    # dimension
nlist = 100     # number of clasters
index = faiss.IndexFlatL2(d)
index.add(all_clothes_preds)


def make_prediction(path2photo):
    image = cv2.imread(path2photo)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_location = face_recognition.face_locations(image)
    if len(face_location):
        face_image = image[face_location[0][0]:face_location[0][2],
                           face_location[0][3]:face_location[0][1]]
        face_image = face_transform(image=face_image)['image'].unsqueeze(0)
        face_image = face_image.to(device)
        face_pred = face_model(face_image)
        face_pred = face_pred.cpu().detach().numpy()
        D, I = index.search(face_pred, 3)
        I = I[0]
        answer_paths = [clothes_paths[ind] for ind in I]
        return answer_paths
    else:
        return False


hello = '''
Hi! Its an AI Bot, which can recommend u fashion clothes based on your face
Please, send to me you photo
'''


bot = telebot.TeleBot("") # Your bot API token

@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    if message.text == '/start':
        bot.reply_to(message, hello)
    else:
        bot.reply_to(message, help)
os.makedirs('tmp_data')

@bot.message_handler(content_types=['photo'])
def get_pred(message):
    file_info = bot.get_file(message.photo[len(message.photo) - 1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    src = 'tmp_data/' + file_info.file_path.split('/')[-1]
    with open(src, "wb") as new_file:
        new_file.write(downloaded_file)

    outputs = make_prediction(src)

    if outputs == False:
        bot.reply_to(message, 'Sorry, but I dont find your face on photo')
    else:
        for path in outputs:
            bot.send_photo(message.chat.id, open(path, 'rb'))

bot.polling(none_stop=True, interval=1)
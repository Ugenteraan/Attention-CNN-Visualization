'''
Visualize the trained model's feature maps.
'''

import os
from tqdm import tqdm
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

from load_dataset import LoadInputImages
from attention_cnn import AttentionCNN
from runtime_args import args


device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'gpu' else 'cpu')
model = AttentionCNN(image_size=args.img_size, image_depth=args.img_depth, num_classes=args.num_classes, drop_prob=args.dropout_rate, device=device)

model = model.to(device)

assert os.path.exists(args.saved_model_name), 'A trained model does not exist!'

try:
    model.load_state_dict(torch.load(args.saved_model_name))
    print("Model loaded!")
except Exception as e:
    print(e)

model.eval()

input_data = LoadInputImages(input_folder=args.data_folder, image_size=args.img_size, image_depth=args.img_depth, transform=transforms.ToTensor())
data_generator = DataLoader(input_data, batch_size=1, shuffle=False, num_workers=1)


output_folder = './output'

if not os.path.exists(output_folder) : os.makedir(output_folder)

for i, image in tqdm(enumerate(data_generator)):

    image = image.to(device)
    image = torch.unsqueeze(image, dim=0)

    attention_filters, cnn_filters, output = model(image)

    attention_combined_filter = cv2.resize(torch.max(attention_filters.squeeze(0), 0)[0].detach().numpy(), (args.img_size, args.img_size))








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

assert os.path.exists(args.model_save_path.rstrip('/')+'/attention_cnn.pth'), 'A trained model does not exist!'

try:
    model.load_state_dict(torch.load(args.model_save_path.rstrip('/')+'/attention_cnn.pth'))
    print("Model loaded!")
except Exception as e:
    print(e)

model.eval()

input_data = LoadInputImages(input_folder=args.data_folder, image_size=args.img_size, image_depth=args.img_depth, transform=transforms.ToTensor())
data_generator = DataLoader(input_data, batch_size=1, shuffle=False, num_workers=1)

class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

output_folder = './output'

if not os.path.exists(output_folder) : os.makedir(output_folder)


fig = plt.figure(figsize=(20, 5))

for i, image in tqdm(enumerate(data_generator)):


    plt.clf()

    image = image.to(device)

    attention_filters, cnn_filters, output = model(image)

    #identify the predicted class
    softmaxed_output = torch.nn.Softmax(dim=1)(output)
    predicted_class = class_names[torch.argmax(softmaxed_output).cpu().numpy()]


    #merge all the filters together as one and resize them to the original image size for viewing.
    attention_combined_filter = cv2.resize(torch.max(attention_filters.squeeze(0), 0)[0].detach().numpy(), (args.img_size, args.img_size))
    cnn_combined_filter = cv2.resize(torch.max(cnn_filters.squeeze(0), 0)[0].detach().numpy(), (args.img_size, args.img_size))


    input_img = cv2.resize(image.squeeze(0).permute(1, 2, 0).cpu().numpy(), (args.img_size, args.img_size))
    #since the filters have a channel dimension of one, we duplicate the filters across all 3 channels to visualize the heatmap with the input images.
    # heatmap_att = cv2.addWeighted(input_img, 0.1,
    #                               np.repeat(np.asarray(np.expand_dims(attention_combined_filter, axis=2), dtype=np.float32), 3, axis=2), 0.8, 0)

    # heatmap_cnn = cv2.addWeighted(input_img, 0.1,
    #                               np.repeat(np.asarray(np.expand_dims(cnn_combined_filter, axis=2), dtype=np.float32), 3, axis=2), 0.8, 0)

    #create heatmap by overlaying the filters on the original image
    heatmap_att = cv2.addWeighted(np.asarray(input_img[:,:, 1], dtype=np.float32), 0.97, np.asarray(attention_combined_filter, dtype=np.float32), 0.07, 0)
    heatmap_cnn = cv2.addWeighted(np.asarray(input_img[:,:, 1], dtype=np.float32), 0.97, np.asarray(cnn_combined_filter, dtype=np.float32), 0.07, 0)

    fig.add_subplot(151)
    plt.imshow(input_img)
    plt.title("Input Image")
    plt.xticks(())
    plt.yticks(())

    fig.add_subplot(152)
    plt.imshow(attention_combined_filter)
    plt.title("Attention Feature Map")
    plt.xticks(())
    plt.yticks(())

    fig.add_subplot(153)
    plt.imshow(cnn_combined_filter)
    plt.title("CNN Feature Map")
    plt.xticks(())
    plt.yticks(())

    fig.add_subplot(154)
    plt.imshow(heatmap_att)
    plt.title("Attention Heat Map")
    plt.xticks(())
    plt.yticks(())

    fig.add_subplot(155)
    plt.imshow(heatmap_cnn)
    plt.title("CNN Heat Map")
    plt.xticks(())
    plt.yticks(())

    fig.suptitle(f"Network's prediction : {predicted_class.capitalize()}", fontsize=20)

    plt.savefig(f'{output_folder}/{i}.png')









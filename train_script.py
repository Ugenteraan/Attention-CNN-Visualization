'''Script to train the Attention-CNN model.
'''

from numpy.lib.type_check import imag
from attention_cnn import AttentionCNN
from load_dataset import LoadDataset
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchsummary import summary
from torchvision import transforms

from runtime_args import args
from load_dataset import LoadDataset

device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'gpu' else 'cpu')


train_dataset = LoadDataset(dataset_folder_path=args.data_folder, image_size=args.img_size, image_depth=args.img_depth, train=True,
                            transform=transforms.ToTensor())
test_dataset = LoadDataset(dataset_folder_path=args.data_folder,image_size=args.img_size, image_depth=args.img_depth, train=False,
                            transform=transforms.ToTensor())

train_generator = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
test_generator = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)


model = AttentionCNN(image_size=args.img_size, image_depth=args.img_depth, num_classes=args.num_classes, drop_prob=args.dropout_rate, device=device)
optimizer = Adam(model.parameters(), lr=args.learning_rate)
criterion = torch.nn.CrossEntropyLoss()

model = model.to(device)
summary(model, (args.img_depth, args.img_size, args.img_size))

best_accuracy = 0
for epoch_idx in range(args.epoch):

    model.train()

    epoch_loss = []
    epoch_accuracy = []
    i = 0
    for i, sample in tqdm(enumerate(train_generator)):

        batch_x, batch_y = sample['image'].to(device), sample['label'].to(device)

        optimizer.zero_grad()

        _,_,net_output = model(batch_x)

        total_loss = criterion(input=net_output, target=batch_y)

        total_loss.backward()
        optimizer.step()
        batch_acc = model.calculate_accuracy(predicted=net_output, target=batch_y)
        epoch_loss.append(total_loss.item())
        epoch_accuracy.append(batch_acc)

    curr_accuracy = sum(epoch_accuracy)/(i+1)
    curr_loss = sum(epoch_loss)/(i+1)

    print(f"Epoch {epoch_idx}")
    print(f"Training Loss : {curr_loss}, Training accuracy : {curr_accuracy}")

    model.eval()

    i = 0
    for i, sample in tqdm(enumerate(test_generator)):

        batch_x, batch_y = sample['image'].to(device), sample['label'].to(device)

        _,_,net_output = model(batch_x)

        total_loss = criterion(input=net_output, target=batch_y)

        batch_acc = model.calculate_accuracy(predicted=net_output, target=batch_y)
        epoch_loss.append(total_loss.item())
        epoch_accuracy.append(batch_acc)

    curr_accuracy = sum(epoch_accuracy)/(i+1)
    curr_loss = sum(epoch_loss)/(i+1)


    print(f"Testing Loss : {curr_loss}, Testing accuracy : {curr_accuracy}")



    if curr_accuracy > best_accuracy:
        torch.save(model.state_dict(), args.model_save_path.rstrip('/')+'/attention_cnn.pth')
        best_accuracy = curr_accuracy
        print("Model is saved!")

    print('------------------------------------------------------------------------------')












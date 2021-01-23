'''
CNN with an attention mechanism.
'''

import torch
import torch.nn as nn


class AttentionCNN(nn.Module):
    '''A CNN arch.
    '''

    def __init__(self, image_size, image_depth, num_classes, drop_prob, device):
        '''Init parameters.
        '''

        super(AttentionCNN, self).__init__()

        self.image_size = image_size
        self.image_depth = image_depth
        self.num_classes = num_classes
        self.drop_prob = drop_prob
        self.device = device

        self.build_model()


    def init_weights(self, m):
        '''Weight initialization for the layers.
        '''

        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def build_model(self):
        '''Build the architecture of the CNN model.
        '''

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=self.image_depth, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.feature_vector_size = (self.image_size//(2**3))**2 * 128

        self.attention = nn.Sequential(
            nn.Linear(self.feature_vector_size, self.feature_vector_size),
            nn.ReLU(inplace=True)
        )
        self.weight = nn.Parameter(torch.randn(1, self.feature_vector_size)*0.05)

        self.fc_layers = nn.Sequential(nn.Linear(self.feature_vector_size, 256),
                                                 nn.ReLU(inplace=True),
                                                 nn.Dropout(p=self.drop_prob),
                                                 nn.Linear(256, self.num_classes))


    def forward(self, x):
        '''Forward Propagation.
        '''

        x = self.conv_layers(x)
        x1 = torch.flatten(x, 1)

        attention_out = nn.Sigmoid()(self.weight * self.attention(x1))
        x1 = attention_out*x1

        reshaped_filters = x1.view(-1, 128, self.image_size//(2**3), self.image_size//(2**3))

        output = self.fc_layers(x1)
        return reshaped_filters, x, output


    def calculate_accuracy(self, predicted, target):
        '''Calculates the accuracy of the prediction.
        '''

        num_data = target.size()[0]
        predicted = torch.argmax(predicted, dim=1)
        correct_pred = torch.sum(predicted == target)

        accuracy = correct_pred*(100/num_data)

        return accuracy.item()
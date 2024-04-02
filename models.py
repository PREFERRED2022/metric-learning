import torch
import torch.nn as nn
import torch.nn.init as init

'''
class Siamese(nn.Module):
    def __init__(self):
        super(Siamese, self).__init__()
        self.embedding_net = embedding_net
        self.model = nn.Sequential(nn.Linear(90,256),nn.ReLU(),nn.Linear(256,512),nn.ReLU(),nn.Linear(512,1024))

    def forward(self,x1,x2,x3):
        out1 = self.model(x1)
        out2 = self.model(x2)
        out3 = self.model(x3)

        return out1, out2, out3
    
    def get_embedding(self, x):
        return self.embedding_net(x)
    '''
class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.model = nn.Sequential(nn.Linear(32,128)
                                   #,nn.BatchNorm1d(256)
                                   ,nn.PReLU()
                                   ,nn.Linear(128,128)
                                   #,nn.BatchNorm1d(256)
                                   ,nn.PReLU()
                                   ,nn.Linear(128,16)
                                   #,nn.BatchNorm1d(16)
        )

    def forward(self,x):
        #print('here1:')
        out = self.model(x)
        return out

    def get_embedding(self, x):
        return self.forward(x)

class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)

class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.linear = nn.Linear(16, 128)
        #self.batch_norm = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128,2)

    def forward(self, x):
        x = self.relu(self.linear(x))
        return torch.softmax(self.linear2(x),dim=1)

class SimpleClassifier1(nn.Module):
    def __init__(self):
        super(SimpleClassifier1, self).__init__()
        self.fc1 = nn.Linear(16, 128)  # Input size is 16
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 2)  # Output size is 10 for 10 classes (0-9)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class EmbeddingNetWithClassifier(nn.Module):
    def __init__(self):
        super(EmbeddingNetWithClassifier, self).__init__()
        self.embedding_net = EmbeddingNet()
        self.triplet_net = TripletNet(self.embedding_net)
        self.classifier = SimpleClassifier1()

    def forward(self, x1, x2, x3):
        embedding_output = self.triplet_net(x1, x2, x3)
        classifier_input = embedding_output[0].view(-1, 16)
        classifier_output = self.classifier(classifier_input)
        return embedding_output, classifier_output

'''    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                # Apply He initialization to Conv2d and Linear layers
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)'''


'''class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(256, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    '''
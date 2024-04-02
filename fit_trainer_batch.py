import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.optim import lr_scheduler
from models import EmbeddingNetWithClassifier
from losses import TripletLoss
import os
from trainer_siam_class_batch import fit
from test_set import testing_results
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.utils.data import RandomSampler
import numpy as np

def visualize_embeddings(model, test_loader, device, dataset, name = None):
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for data in test_loader:
            anchor, anchor_label = data[0]
            positive, positive_label = data[1]
            negative, negative_label = data[2]
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            # Get the embeddings for the anchor samples
            embedding1 = model.embedding_net.get_embedding(anchor.float())
            embeddings.append(embedding1.cpu().numpy())
            embedding2 = model.embedding_net.get_embedding(positive.float())
            embeddings.append(embedding2.cpu().numpy())
            embedding3 = model.embedding_net.get_embedding(negative.float())
            embeddings.append(embedding3.cpu().numpy())
            labels.extend(anchor_label)
            labels.extend(positive_label)
            labels.extend(negative_label)

    embeddings = np.concatenate(embeddings)
    labels = np.array(labels)

    # Perform t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot three subplots based on labels
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))

    # Plot all samples
    scatter_all = axs[0].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis', s=10)
    axs[0].set_title('All Samples')
    axs[0].set_xlabel('Dimension 1')
    axs[0].set_ylabel('Dimension 2')

    # Plot samples with label 0
    scatter_label_0 = axs[1].scatter(embeddings_2d[labels == 0, 0], embeddings_2d[labels == 0, 1], c='blue', s=10)
    axs[1].set_title('Samples with Label 0')
    axs[1].set_xlabel('Dimension 1')
    axs[1].set_ylabel('Dimension 2')

    # Plot samples with label 1
    scatter_label_1 = axs[2].scatter(embeddings_2d[labels == 1, 0], embeddings_2d[labels == 1, 1], c='orange', s=10)
    axs[2].set_title('Samples with Label 1')
    axs[2].set_xlabel('Dimension 1')
    axs[2].set_ylabel('Dimension 2')

    # Adjust layout
    plt.tight_layout()
    plt.suptitle(f't-SNE Visualization of Embeddings ({dataset})', y=1.05)
    if name is not None:
        plt.savefig(name)
    else:
        plt.show()


torch.manual_seed(0)
cuda = torch.cuda.is_available()
device = torch.device("cuda:0")
#par_path = '/home/sgirtsou/Documents/siamese_datasets/n_dataset_4/new_m_manhattan_e1_sh0_norm_fanc10_nfanc2_r1'
#par_path = '/home/sgirtsou/Documents/siamese_datasets/n_dataset_4/m_manhattan_e1_sh0_norm_fanc10_nfanc1_r0.7'
#par_path = '/home/sgirtsou/Documents/siamese_datasets/n_dataset_4/experiments/easy7_semihard3_n10'
#par_path = '/home/sgirtsou/Documents/siamese_datasets/n_dataset_4/experiments/m_manhattan_e6.0_se2.0_norm_nf10_nnf1_r0.5/'
#par_path = '/home/sgirtsou/Documents/siamese_datasets/n_dataset_6/experiments/m_manhattan_pn10.0_0.0_0.0_np10_0_0_norm_nf10_nnf10_r0.9'
par_path = '/home/sgirtsou/Documents/siamese_datasets/n_dataset_7/experiments/m_manhattan_pn7_1_1_np2_1_1_norm_nf20_nnf1.0_r0.9/'
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}


train_batch_sampler = torch.load(os.path.join(par_path, 'train_batch_sampler.pth'))
#test_batch_sampler = torch.load(os.path.join(par_path, 'test_batch_sampler.pth'))

#shuffled_sampler = RandomSampler(train_batch_sampler)

total_batches = len(train_batch_sampler)

# Decide the split ratio (e.g., 80% for training, 20% for testing)
train_ratio = 0.8
train_size = int(total_batches * train_ratio)
test_size = total_batches - train_size

# Split the 'train_batch_sampler' into 'train_sampler' and 'test_sampler'
train_batch_sampler, test_batch_sampler = torch.utils.data.random_split(train_batch_sampler, [train_size, test_size])

train_loader = torch.utils.data.DataLoader(train_batch_sampler, batch_size=50, **kwargs)
test_loader = torch.utils.data.DataLoader(test_batch_sampler, batch_size=50, **kwargs)

margin = 0.1
model = EmbeddingNetWithClassifier()
loss_fn = TripletLoss(margin)
lr = 0.001
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-5) #ip wheight_decay maybe big --5
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.5, last_epoch=-1)
n_epochs = 2
log_interval = 200
start_epoch = 0
lambda_cls = 50
class_weights=[1,4]

fit(train_loader, test_loader,  model, loss_fn, optimizer, scheduler, n_epochs, par_path=None, start_epoch=0, lambda_cls=lambda_cls, class_weights=class_weights)

#visualize_embeddings(model,train_loader,device,dataset='training')
#visualize_embeddings(model, test_loader, device,dataset='test')
specificity,sensitivity = testing_results(model,device,year='2020',months = ['08','09','06','07'])

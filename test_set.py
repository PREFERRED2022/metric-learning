import json
import pandas as pd
from datasets import myDataset, CustomDatasetClassification
import json
from sklearn.metrics import f1_score, confusion_matrix
import glob
import os
import torch
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np

def visualize_embeddings_triplets(model, test_loader, device, dataset, name = None):
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

def visualize_embeddings(model, embeddings, labels, device, dataset):
    model.eval()

    # Perform t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings.cpu().detach().numpy())

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
    plt.show()

def normalization_and_ohe(df,normalization = False, ohe = True):
    if normalization == True:
        with open('/mnt/nvme2tb/ffp/datasets/norm_values_ref_final.json', 'r') as file:
            norm_data = json.load(file)

        for key in norm_data.keys():
            try:
                df[key] = (df[key] - norm_data[key]['min']) / (norm_data[key]['max'] - norm_data[key]['min'])
            except:
                continue
    else:
        i=1
        #print('No normalization applied to these data')

    #print('Applying one hot encoding')
    cat_vars = {'dom_dir': [1, 2, 3, 4, 5, 6, 7, 8],
                'dir_max': [1, 2, 3, 4, 5, 6, 7, 8],
                'weekday': [1, 2, 3, 4, 5, 6, 7],
                'month': [3, 4, 5, 6, 7, 8, 9, 10]}
    for key, value in cat_vars.items():
        categorical_variable = key
        if ohe == True:
            df[categorical_variable] = df[categorical_variable].astype('int')
            y = pd.get_dummies(df[categorical_variable], prefix=categorical_variable)
            df = df.join(y)
        try:
            df = df.drop(categorical_variable, axis=1)
        except:
            continue
    return df

def preprocessing_df(file_path,df_train, ohe_user = True):
    df = pd.read_csv(file_path)
    columns_to_delete = ['corine_13', 'corine_4', 'corine_30', 'corine_38', 'corine_36', 'corine_10', 'corine_20',
                         'corine_18', 'corine_16',
                         'corine_43', 'corine_27', 'corine_5', 'corine_31', 'corine_17', 'corine_12', 'corine_26',
                         'corine_37', 'corine_24',
                         'corine_28', 'corine_29', 'corine_19', 'corine_21', 'corine_6', 'corine_23', 'corine_22',
                         'corine_1', 'corine_8',
                         'corine_33', 'corine_2', 'corine_40', 'corine_32', 'corine_41', 'corine_3', 'corine_25',
                         'corine_35', 'corine_14',
                         'corine_42', 'corine_11', 'corine_44', 'corine_15', 'corine_7', 'corine_9', 'index', 'id',
                         'firedate', 'band']
    for col in columns_to_delete:
        try:
            df = df.drop(col, axis=1)
        except:
            continue
    #print("Number of rows:", df.shape[0])
    #print('Read df successfully')
    df = df.dropna()

    df = normalization_and_ohe(df, normalization=False, ohe = ohe_user)
    for col_name in list(set(df_train.columns) - set(df.columns)):
        df[col_name] = 0

    df = df[df_train.columns]
    df = df.replace({False: 0, True: 1})
    return df
def testing_results(model,device,year='2019', months = ['08'], ohe_user = False):
    confusion_mat = np.zeros((2, 2))
    directory_path = f'/mnt/nvme2tb/ffp/datasets/test/{year}/greece/'
    df_train = pd.read_csv(os.path.join('/home/sgirtsou/Documents/siamese_datasets/n_dataset_7/fires.csv'))
    df_train = normalization_and_ohe(df_train, normalization=False, ohe=ohe_user)
    file_paths = []
    for month in months:
        print(month)
        file_pattern = f'{year}{month}??_df_greece_norm.csv'
        #file_pattern = f'2019{month}??_df_norm.csv'
        # Use glob to get a list of file paths that match the pattern
        month_paths = glob.glob(directory_path + file_pattern)
        file_paths.append(month_paths)

    file_paths = [item for sublist in file_paths for item in sublist]

    # Iterate through the file paths and read each CSV file into a DataFrame
    for file_path in file_paths:
        print(file_path)
        df = preprocessing_df(file_path, df_train, ohe_user=ohe_user)
        df_data = df.drop(columns='fire').values
        df_labels = df.fire.values
        # Get unique values and their counts
        '''unique_values, counts = np.unique(df_labels, return_counts=True)
        # Create a dictionary to store the frequencies
        frequency_dict = dict(zip(unique_values, counts))
        # Print the result
        for value, frequency in frequency_dict.items():
            print(f"Value: {value}, Frequency: {frequency}")'''
        df_val = CustomDatasetClassification(df_data,df_labels)
        del df
        val_class_loader = torch.utils.data.DataLoader(df_val, batch_size=500, shuffle=True, num_workers=15, drop_last = True)
        del df_val
        total_batches = len(val_class_loader)
        progress_bar = tqdm(total=total_batches)

        all_predictions = []
        all_labels = []
        all_embeddings = []
        with (torch.no_grad()):
            for i, data in enumerate(val_class_loader):
                progress_bar.update(1)
                model.eval()
                image = data['data']
                target = data['label']
                image = image.to(device)
                try:
                    model1 = model.embedding_net.to(device)
                except:
                    model1 = model.to(device)
                embeddings = model1.get_embedding(image.float())
                all_embeddings.append(embeddings)
                labels = target.to(device)
                val_embeddings = embeddings.to(device)
                val_outputs = model.classifier(val_embeddings)
                _, predicted_labels = torch.max(val_outputs, 1)

                labels = labels.cpu().numpy().ravel()
                all_predictions.extend(predicted_labels.cpu())
                all_labels.extend(labels.ravel())
            progress_bar.close()
        # Convert lists to NumPy arrays
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        #all_embeddings = torch.cat(all_embeddings, dim=0)
        #visualize_embeddings(model, all_embeddings, all_labels, device, file_path.split('/')[-1])
        # Update the confusion matrix
        conf_matrix = confusion_matrix(all_labels, all_predictions)
        confusion_mat += conf_matrix
        print(f'Confusion mat for {file_path}:{conf_matrix}')

    print('Prediction completed')

    tn, fp, fn, tp = confusion_mat.ravel()
    # Calculate specificity and sensitivity
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)

    # print("Epoch {}: Conf matrix = {:.4f}".format(epoch + 1, confusion_mat))
    print("Specificity: {:.4f}".format(specificity))
    print("Sensitivity: {:.4f}".format(sensitivity))
    return specificity,sensitivity

'''device = torch.device("cuda:0")
file = 'm_manhattan_pn7_1_1_np2_1_1_norm_nf20_nnf1.0_r0.9'
path = '/home/sgirtsou/Documents/siamese_datasets/n_dataset_7/experiments/'
model_name = 'model_18.pth'
model = torch.load(os.path.join(path,file,model_name))
model.eval()
specificity_2021, sensitivity_2021 = testing_results(model, device, year='2021', months=['06','07','08','09'], ohe_user=False)'''
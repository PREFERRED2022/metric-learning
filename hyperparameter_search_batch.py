import torch
from torch.utils.tensorboard import SummaryWriter
cuda = torch.cuda.is_available()
device = torch.device("cuda:0")
import torch.optim as optim
from torch.optim import lr_scheduler
from models import EmbeddingNetWithClassifier
from losses import TripletLoss
import os
import pandas as pd
from trainer_siam_class_batch import fit
from test_set import testing_results
import random

def all_tests():
    # Set of possible values for hyperparameters
    possible_values = {
        'margin': [0.1, 0.3, 0.5, 0.7, 1],
        'lr': [1e-3, 5e-4, 1e-4],
        'gamma': [0.1, 0.5, 0.9],
        'weight_decay': [1e-4, 5e-5], #[1e-4, 5e-5, 1e-5],
        'n_epochs': [10, 20, 30], #[10, 20, 30, 40, 50],
        'lambda_cls': [0, 10, 50, 75, 100],
        'class_weights': [[1, 2], [1, 4]]#, [1, 6]]
    }
    #create a list of all possible combinations of hyperparameters
    hyperparameter_combinations = []
    for margin in possible_values['margin']:
        for lr in possible_values['lr']:
            for gamma in possible_values['gamma']:
                for weight_decay in possible_values['weight_decay']:
                    for n_epochs in possible_values['n_epochs']:
                            for lambda_cls in possible_values['lambda_cls']:
                                for class_weights in possible_values['class_weights']:
                                    hyperparameter_combinations.append(
                                        [margin, lr, gamma, weight_decay, n_epochs, lambda_cls, class_weights])
    return hyperparameter_combinations


# Create a function that takes a list of hyperparameters as input and returns the average validation loss
def fit_hyperparameter_search(hyperparameters,train_loader, test_loader):
    margin, lr, gamma, weight_decay, n_epochs, lambda_cls, class_weights = hyperparameters
    model = EmbeddingNetWithClassifier()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=gamma, last_epoch=-1)
    loss_fn = TripletLoss(margin)
    fit(train_loader, test_loader, model, loss_fn, optimizer, scheduler, n_epochs, start_epoch=0, lambda_cls=lambda_cls,
        class_weights=class_weights)
    specificity,sensitivity = testing_results(model, device, year='2019',months = ['08'])
    return specificity,sensitivity, model

def dataset_search(folder, kwargs):
    try:
        results_csv = pd.read_csv(os.path.join(folder, 'results.csv'))
        mode = 'created'
    except:
        mode = 'not_created'
        pass
    for i, par_path in enumerate(os.listdir(folder)):
        results = []
        if mode == 'created':
            if par_path in results_csv['Dataset'].values:
                print(f'{par_path} already tested')
                continue
        if os.path.isdir(os.path.join(folder,par_path)):
            print(f'Testing {par_path} {i+1}/{len(os.listdir(folder))}')
            try:
                train_batch_sampler = torch.load(os.path.join(folder, par_path, 'train_batch_sampler.pth'))
                total_batches = len(train_batch_sampler)
                # Decide the split ratio (e.g., 80% for training, 20% for testing)
                train_ratio = 0.8
                train_size = int(total_batches * train_ratio)
                test_size = total_batches - train_size
                # Split the 'train_batch_sampler' into 'train_sampler' and 'test_sampler'
                train_batch_sampler, test_batch_sampler = torch.utils.data.random_split(train_batch_sampler, [train_size, test_size])
                train_loader = torch.utils.data.DataLoader(train_batch_sampler, batch_size=50, **kwargs)
                test_loader = torch.utils.data.DataLoader(test_batch_sampler, batch_size=50, **kwargs)
                model = EmbeddingNetWithClassifier()
                margin = 0.3
                loss_fn = TripletLoss(margin)
                optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-5)
                scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
                n_epochs = 12
                fit(train_loader, test_loader, model, loss_fn, optimizer, scheduler, n_epochs, start_epoch=0,lambda_cls=100,
                    class_weights=[1,6])
                specificity_2019, sensitivity_2019 = testing_results(model, device, year='2019', months=['08'], ohe_user=False)
                specificity_2020, sensitivity_2020 = testing_results(model, device, year='2020', months=['08'], ohe_user=False)
                #if i == 0 and not os.path.exists(os.path.join(folder, 'results.csv')):
                if mode == 'not_created':
                    results.append([par_path, specificity_2019, sensitivity_2019, specificity_2020, sensitivity_2020])
                    results = pd.DataFrame(results, columns=['Dataset', 'Specificity_2019', 'Sensitivity_2019',
                                                             'Specificity_2020', 'Sensitivity_2020'])
                    results.to_csv(os.path.join(folder, 'results.csv'), index=False)
                    mode = 'created_now'
                else:
                    results.append([par_path, specificity_2019, sensitivity_2019, specificity_2020, sensitivity_2020])
                    results = pd.DataFrame(results, columns=['Dataset', 'Specificity_2019', 'Sensitivity_2019',
                                                             'Specificity_2020', 'Sensitivity_2020'])
                    results.to_csv(os.path.join(folder, 'results.csv'), mode='a', header=False, index=False)
            except:
                continue


def hyperparameter_search(hyperparameter_combinations, train_loader, test_loader, par_path):
    #random select 400 combinations of hyperparameters
    random.seed(30)
    hyperparameter_combinations = random.sample(hyperparameter_combinations, 400)
    full_tests = 0
    # Run the hyperparameter search and save the results in a csv file
    for i, hyperparameters in enumerate(hyperparameter_combinations):
        results = []
        print(f'Hyperparameter combination {i + 1}/{len(hyperparameter_combinations)}')
        specificity, sensitivity, model = fit_hyperparameter_search(hyperparameters, train_loader, test_loader)
        if sensitivity > 0.6 and specificity > 0.6: # and full_tests < 20:
            model_name = f'model_{i}.pth'
            torch.save(model, os.path.join(par_path, model_name))
            specificity_2019, sensitivity_2019 = testing_results(model, device, year='2019', months=['08', '07', '06', '09'])
            specificity_2020, sensitivity_2020 = testing_results(model, device, year='2020', months=['08', '07', '06', '09'])
            full_tests += 14
        else:
            specificity_2019, sensitivity_2019 = 0, 0
            specificity_2020, sensitivity_2020 = 0, 0
            model_name = 'None'
        if i == 0:
            results.append([hyperparameters, specificity, sensitivity, specificity_2019, sensitivity_2019, specificity_2020, sensitivity_2020, model_name])
            results = pd.DataFrame(results, columns=['Hyperparameters', 'Specificity', 'Sensitivity', 'Specificity_2019', 'Sensitivity_2019','Specificity_2020', 'Sensitivity_2020', 'Model_name'])
            results.to_csv(os.path.join(par_path, 'results.csv'), index=False)
        else:
            results.append([hyperparameters, specificity, sensitivity,  specificity_2019, sensitivity_2019, specificity_2020, sensitivity_2020, model_name])
            results = pd.DataFrame(results, columns=['Hyperparameters', 'Specificity', 'Sensitivity', 'Specificity_2019', 'Sensitivity_2019','Specificity_2020', 'Sensitivity_2020', 'Model_name'])
            results.to_csv(os.path.join(par_path, 'results.csv'), index=False, mode='a', header=False)

if __name__ == '__main__':
    folder = '/home/sgirtsou/Documents/siamese_datasets/n_dataset_7/experiments/'
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    file = 'm_manhattan_pn8_2_2_np2_1_1_norm_nf20_nnf1.0_r0.7'
    par_path = os.path.join(folder, file)
    # run multiple tests with different hyperparameters
    # for each test, save the results in a csv file
    
    train_batch_sampler = torch.load(os.path.join(par_path, 'train_batch_sampler.pth'))
    total_batches = len(train_batch_sampler)
    train_ratio = 0.8
    train_size = int(total_batches * train_ratio)
    test_size = total_batches - train_size
    train_batch_sampler, test_batch_sampler = torch.utils.data.random_split(train_batch_sampler,
                                                                            [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_batch_sampler, batch_size=50, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_batch_sampler, batch_size=50, **kwargs)
    hyperparameter_combinations = all_tests()
    hyperparameter_search(hyperparameter_combinations, train_loader, test_loader, par_path)

    #dataset_search(folder,kwargs)
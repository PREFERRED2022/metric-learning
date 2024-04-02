import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np
from test_set import visualize_embeddings_triplets

def fit(train_loader, test_loader,  model, loss_fn, optimizer, scheduler, n_epochs, par_path = None, start_epoch=0,lambda_cls=50, class_weights=[1,4]):
    device = torch.device("cuda:0")
    for epoch in range(0, start_epoch):
        scheduler.step()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/trainer_sim_class_{}'.format(timestamp))
    best_vloss = 1_000_000.
    model.to(device)

    for epoch_number in range(start_epoch, n_epochs):
        print('EPOCH {}:'.format(epoch_number + 1))
        model.train(True)
        avg_loss_triplet, avg_loss_class, avg_loss = train_one_epoch(epoch_number, train_loader, optimizer, model, device, loss_fn, lambda_cls, writer,class_weights)
        running_vloss = 0.0
        scheduler.step()
        model.eval()
        avg_vloss_triplet, avg_vloss_class, avg_vloss = test_one_epoch(test_loader,model, device, loss_fn, lambda_cls, class_weights)

        if (epoch_number + 1) % 5 == 0 and par_path is not None:
            visualize_embeddings_triplets(model, train_loader, device, dataset='training', name = f'{par_path}+train_{str(epoch_number+1)}.png')
            visualize_embeddings_triplets(model, test_loader, device,dataset='test', name = f'{par_path}+test_{str(epoch_number+1)}.png')

        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Triplet Loss',
                           {'Training': avg_loss_triplet, 'Validation': avg_vloss_triplet},
                           epoch_number + 1)
        writer.add_scalars('Training vs. Validation Class Loss',
                           {'Training': avg_loss_class, 'Validation': avg_vloss_class},
                           epoch_number + 1)
        writer.add_scalars('Training vs. Validation Total Loss',
                           {'Training': avg_loss, 'Validation': avg_vloss},
                           epoch_number + 1)
        writer.flush()
'''
        # Track the best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'model_{}_{}'.format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)

'''
def train_one_epoch(epoch_index, train_loader, optimizer, model, device, loss_fn,lambda_cls, tb_writer, class_weights):
    running_loss = 0.
    last_loss = 0.
    running_loss_triplet = 0.0
    running_loss_class = 0.0
    running_total_loss = 0.0
    model_triplet = model.triplet_net.to(device)
    emb_net = model.embedding_net.to(device)
    class_net = model.classifier.to(device)
    weight_tensor = torch.Tensor(class_weights).to(device)
    classifier_loss_fn = torch.nn.CrossEntropyLoss(weight_tensor, reduction='mean')

    for i, data in enumerate(train_loader):
        anchor, anchor_label = data[0]
        positive, positive_label = data[1]
        negative, negative_label = data[2]
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        optimizer.zero_grad()
        out1, out2, out3 = model_triplet(anchor.float(), positive.float(), negative.float())
        outputs = [out1, out2, out3]
        target = [anchor_label, positive_label, negative_label]

        loss_outputs = loss_fn(*outputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs

        c_out1 = emb_net.get_embedding(anchor.float())
        c_out2 = emb_net.get_embedding(positive.float())
        c_out3 = emb_net.get_embedding(negative.float())

        classifier_input = torch.cat((c_out1, c_out2, c_out3), dim=0)

        numpy_class = classifier_input.cpu().detach().numpy()
        unique_rows, unique_indices = np.unique(numpy_class, axis=0, return_index=True)
        target_np = np.concatenate(target)
        target_class = target_np[unique_indices]
        target_torch = torch.from_numpy(target_class).to(device)
        classifier_input = torch.from_numpy(unique_rows).to(device)

        classifier_output = class_net(classifier_input)
        classifier_loss = classifier_loss_fn(classifier_output, target_torch)

        total_loss = loss + lambda_cls * classifier_loss
        #print(f'Total loss is {total_loss}')
        #print(f'Loss item: {loss.item()}, Classifier loss item: {classifier_loss.item()}')
        total_loss.backward()
        #loss.backward()
        optimizer.step()
        # Gather data and report
        running_total_loss += total_loss
        running_loss += loss.item()
        running_loss_triplet += loss
        running_loss_class += classifier_loss

        #return the loss of embedding net and of classifier to monitor them in tensorboard
        if i % 1000 == 999:
            last_loss = running_loss / 1000  # loss per batch
            last_triplet_loss = running_loss_triplet/1000
            last_class_loss = running_loss_class/1000
            print('TRAINING')
            print('batch {} loss: {}'.format(i + 1, total_loss))
            print('batch {} triplet loss: {}'.format(i + 1, last_triplet_loss))
            print('batch {} classification loss: {}'.format(i + 1, last_class_loss))
            tb_x = epoch_index * len(train_loader) + i + 1
            tb_writer.add_scalar('Loss/train', total_loss, tb_x)
            tb_writer.add_scalar('Triplet Loss/train', last_triplet_loss, tb_x)
            tb_writer.add_scalar('Class Loss/train', last_class_loss, tb_x)
            running_loss = 0.

    avg_loss_triplet = running_loss_triplet / (i + 1)
    avg_loss_class = running_loss_class / (i + 1)
    avg_total_loss = running_total_loss / (i+1)
    #print(f'Training: Avg Triplet loss: {avg_loss_triplet.item()}, Avg classifier loss: {avg_loss_class.item()}')
    return avg_loss_triplet, avg_loss_class, avg_total_loss

def test_one_epoch(test_loader,model,device, loss_fn,lambda_cls, class_weights):
    running_vloss_triplet = 0.0
    running_vloss_class = 0.0
    running_total_vloss = 0.0
    model_triplet = model.triplet_net.to(device)
    emb_net = model.embedding_net.to(device)
    class_net = model.classifier.to(device)
    weight_tensor = torch.Tensor(class_weights).to(device)
    classifier_loss_fn = torch.nn.CrossEntropyLoss(weight_tensor)
    model.eval()
    with torch.no_grad():
        for i, vdata in enumerate(test_loader):
            anchor, anchor_label = vdata[0]
            positive, positive_label = vdata[1]
            negative, negative_label = vdata[2]
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            out1, out2, out3 = model.triplet_net.forward(anchor.float(), positive.float(), negative.float())
            outputs = [out1, out2, out3]
            target = [anchor_label, positive_label, negative_label]
            loss_outputs = loss_fn(*outputs)
            vloss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs

            c_out1 = emb_net.get_embedding(anchor.float())
            c_out2 = emb_net.get_embedding(positive.float())
            c_out3 = emb_net.get_embedding(negative.float())

            classifier_input = torch.cat((c_out1, c_out2, c_out3), dim=0)

            numpy_class = classifier_input.cpu().detach().numpy()
            unique_rows, unique_indices = np.unique(numpy_class, axis=0, return_index=True)
            target_np = np.concatenate(target)
            target_class = target_np[unique_indices]
            target_torch = torch.from_numpy(target_class).to(device)
            classifier_input = torch.from_numpy(unique_rows).to(device)

            classifier_output = class_net(classifier_input)
            classifier_vloss = classifier_loss_fn(classifier_output, target_torch)

            total_vloss = vloss + lambda_cls * classifier_vloss
            running_vloss_triplet += vloss
            running_vloss_class += classifier_vloss

            running_total_vloss += total_vloss

            if i % 400 == 399:
                last_triplet_vloss = running_vloss_triplet / 400
                last_class_vloss = running_vloss_class / 400
                print('VALIDATION')
                print('batch {} Vloss: {}'.format(i + 1, total_vloss))
                print('batch {} triplet Vloss: {}'.format(i + 1, last_triplet_vloss))
                print('batch {} classification Vloss: {}'.format(i + 1, last_class_vloss))

    print(i)

    avg_vloss_triplet = running_vloss_triplet / (i + 1)
    avg_vloss_class = running_vloss_class / (i + 1)
    avg_total_vloss = running_total_vloss/ (i + 1)

    return avg_vloss_triplet, avg_vloss_class, avg_total_vloss
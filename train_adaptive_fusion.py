# -*- coding: utf-8 -*-
"""
Training script for Hybrid LIF-GNN with Adaptive Fusion

Implements training loop with support for multiple fusion strategies:
- Simple: Learnable scalar weight
- Gated: Feature-based gating
- Attention: Cross-attention fusion
"""

from __future__ import print_function
import os
import time
from datetime import date
import numpy as np
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset, DataLoader
from tactileSGNet_adaptive_fusion import *
from torch.autograd import Variable
import random
import tqdm
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import json

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
random.seed(0)

# Hyperparameter setting
parser = argparse.ArgumentParser("Train TactileSGNet with Adaptive Fusion")
parser.add_argument('--sample_file', type=int, default=1, help='Sample file number')
parser.add_argument('--fusion_type', type=str, default='gated', 
                    choices=['simple', 'gated', 'attention','classwise','concat' ,'entropy'],
                    help='Type of fusion mechanism')
parser.add_argument('--num_classes', type=int, default=36, help='Number of classes')
parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--lr_decay_epoch', type=int, default=40, help='LR decay epoch')
parser.add_argument('--use_intermediate_fusion', action='store_true', 
                    help='Use intermediate feature fusion')

args = parser.parse_args()

# Model name
model_name = f'_TactileSGNet_AdaptiveFusion_{args.fusion_type}_'

# Hyperparameters
num_classes = args.num_classes
k = 3
num_run = 1
learning_rate = args.learning_rate
num_epochs = args.num_epochs
batch_size = args.batch_size
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
time_length = 250

print("="*60)
print(f"Training Configuration:")
print(f"  Fusion Type: {args.fusion_type}")
print(f"  Num Classes: {num_classes}")
print(f"  Batch Size: {batch_size}")
print(f"  Learning Rate: {learning_rate}")
print(f"  Num Epochs: {num_epochs}")
print(f"  Device: {device}")
print(f"  Intermediate Fusion: {args.use_intermediate_fusion}")
print("="*60)


# Event-based tactile dataset
class tactileDataset(Dataset):
    def __init__(self, data_path, train=True):
        if train:
            self.files = os.listdir(data_path + '/train')
            self.file_path = data_path + '/train/'
        else:
            self.files = os.listdir(data_path + '/test')
            self.file_path = data_path + '/test/'

    def __getitem__(self, index):
        fileName = self.files[index]
        nameStr = fileName.split('_label_')
        label = int(nameStr[-1].split('.')[0])
        data = torch.from_numpy(np.load(self.file_path + fileName))
        label = torch.LongTensor([label])
        return data, label

    def __len__(self):
        return len(self.files)


# Decay learning rate
def lr_scheduler(optimizer, epoch, init_lr=0.01, lr_decay_epoch=40):
    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            if param_group['lr'] <= 1e-5:
                break
            param_group['lr'] = param_group['lr'] * 0.1
            print(f"Learning rate decayed to: {param_group['lr']}")
    return optimizer


def plot_training_curves(acc_list, training_loss_list, test_loss_list, save_path):
    """Plot and save training curves"""
    epochs = range(1, len(acc_list[0]) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    for i, acc in enumerate(acc_list):
        axes[0].plot(epochs, acc.numpy(), label=f'Run {i+1}')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Test Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot losses
    for i, (train_loss, test_loss) in enumerate(zip(training_loss_list, test_loss_list)):
        axes[1].plot(epochs, train_loss.numpy(), label=f'Train Loss Run {i+1}', linestyle='--')
        axes[1].plot(epochs, test_loss.numpy(), label=f'Test Loss Run {i+1}')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Training and Test Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Training curves saved to {save_path}")


def analyze_fusion_weights(fusion_weights_history, save_path):
    """Analyze and plot fusion weight evolution"""
    epochs = range(1, len(fusion_weights_history) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, fusion_weights_history, marker='o', linestyle='-', linewidth=2)
    plt.axhline(y=0.5, color='r', linestyle='--', label='Equal weight (0.5)')
    plt.xlabel('Epoch')
    plt.ylabel('Spatial Stream Weight (α)')
    plt.title('Fusion Weight Evolution During Training')
    plt.legend()
    plt.grid(True)
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Fusion weight analysis saved to {save_path}")


if __name__ == '__main__':
    # Dataset paths
    if num_classes == 20:
        data_path = '../../Ev-Containers'
        print("Dataset: ../Ev-Container")
    elif num_classes == 36:
        data_path = '../../Ev-Objects/'
        print("Dataset: ../Ev-Objects/")
    else:
        print("error: num_classes")
        exit()
    
    trainDataset = tactileDataset(data_path, train=True)
    testDataset = tactileDataset(data_path, train=False)
    train_loader = DataLoader(dataset=trainDataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(dataset=testDataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Training samples: {len(trainDataset)}")
    print(f"Testing samples: {len(testDataset)}")
    
    # Run for num_run times
    best_acc = torch.zeros(num_run)
    acc_list = list([])
    training_loss_list = list([])
    test_loss_list = list([])
    fusion_weights_list = list([])
    net_list = list([])

    for run in range(num_run):
        print(f"\n{'='*60}")
        print(f"Starting Run {run + 1}/{num_run}")
        print(f"{'='*60}")
        
        # Initialize model
        if args.use_intermediate_fusion:
            model = HybridLIFGNN_AdaptiveFusion_WithIntermediate(
                num_classes=num_classes, 
                time_length=time_length, 
                k=k, 
                device=device, 
                sparse=True,
                fusion_type=args.fusion_type
            )
        else:
            model = HybridLIFGNN_AdaptiveFusion(
                num_classes=num_classes, 
                time_length=time_length, 
                k=k, 
                device=device, 
                sparse=True,
                fusion_type=args.fusion_type
            )
        
        model.to(device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)

        acc = torch.zeros(num_epochs)
        training_loss = torch.zeros(num_epochs)
        test_loss = torch.zeros(num_epochs)
        fusion_weights_history = []
        
        best_acc_this_run = 0.0
        best_epoch = 0
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Training
            model.train()
            running_loss = 0
            train_fusion_weights = []
            
            for trainData, trainLabel in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
                model.zero_grad()
                optimizer.zero_grad()
                trainData = trainData.to(device)
                outputs = model(trainData)

                labels_ = torch.zeros(trainLabel.shape[0], num_classes).scatter_(1, trainLabel.view(-1, 1), 1)
                loss = criterion(outputs.cpu(), labels_)

                running_loss += loss.item()
                loss.backward()
                optimizer.step()
                
                # Track fusion weight
                if hasattr(model, 'last_fusion_weight'):
                    train_fusion_weights.append(model.last_fusion_weight)

            training_loss[epoch] = running_loss
            avg_fusion_weight = np.mean(train_fusion_weights) if train_fusion_weights else 0.5
            fusion_weights_history.append(avg_fusion_weight)

            # Testing
            model.eval()
            correct = 0
            total = 0
            optimizer = lr_scheduler(optimizer, epoch, learning_rate, args.lr_decay_epoch)
            running_loss = 0
            predictions_all = []
            labels_all = []
            
            with torch.no_grad():
                for testData, testLabel in tqdm.tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Test]"):
                    outputs = model(testData, False)
                    labels_ = torch.zeros(testLabel.shape[0], num_classes).scatter_(1, testLabel.view(-1, 1), 1)
                    loss = criterion(outputs.cpu(), labels_)
                    running_loss += loss.item()
                    _, predicted = outputs.cpu().max(1)
                    predicted = predicted.unsqueeze(1)
                    
                    total += float(testLabel.size(0))
                    correct += float(predicted.eq(testLabel).sum().item())
                    
                    predictions_all.extend(predicted.numpy().flatten())
                    labels_all.extend(testLabel.numpy().flatten())

            test_loss[epoch] = running_loss
            acc[epoch] = 100. * float(correct) / float(total)
            
            if best_acc[run] < acc[epoch]:
                best_acc[run] = acc[epoch]
                best_acc_this_run = acc[epoch]
                best_epoch = epoch + 1

            # Print progress
            if (epoch + 1) % 2 == 0:
                print(f'\nEpoch: {epoch + 1:3d} | '
                      f'Train Loss: {training_loss[epoch]:8.4f} | '
                      f'Test Loss: {test_loss[epoch]:8.4f} | '
                      f'Acc: {acc[epoch]:6.2f}% | '
                      f'Best: {best_acc[run]:6.2f}% | '
                      f'Fusion α: {avg_fusion_weight:.4f} | '
                      f'Time: {time.time() - start_time:.2f}s')

        print(f"\n{'='*60}")
        print(f"Run {run + 1} completed!")
        print(f"Best Accuracy: {best_acc[run]:.3f}% at epoch {best_epoch}")
        print(f"Final Fusion Weight: {fusion_weights_history[-1]:.4f}")
        print(f"{'='*60}")

        test_loss_list.append(test_loss)
        training_loss_list.append(training_loss)
        acc_list.append(acc)
        fusion_weights_list.append(fusion_weights_history)
        net_list.append(model.state_dict())
        
        # Compute confusion matrix for last epoch
        cm = confusion_matrix(labels_all, predictions_all)
        
        # Save confusion matrix
        if not os.path.isdir('results'):
            os.mkdir('results')
        np.save(f'./results/confusion_matrix_{args.fusion_type}_run{run+1}.npy', cm)

    # Overall statistics
    state = {
        'net_list': net_list,
        'best_acc': best_acc,
        'num_epochs': num_epochs,
        'acc_list': acc_list,
        'training_loss_list': training_loss_list,
        'test_loss_list': test_loss_list,
        'fusion_weights_list': fusion_weights_list,
        'fusion_type': args.fusion_type,
        'hyperparameters': {
            'num_classes': num_classes,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'k': k,
        }
    }
    
    dateStr = date.today().strftime("%Y%m%d")
    
    if not os.path.isdir('log_data_adaptive_fusion'):
        os.mkdir('log_data_adaptive_fusion')
    
    save_path = f'./log_data_adaptive_fusion/{dateStr}{model_name}objects_{num_classes}_k_{k}_file_{args.sample_file}.t7'
    torch.save(state, save_path)
    print(f'\nModel saved to: {save_path}')
    
    # Save summary statistics as JSON
    summary = {
        'fusion_type': args.fusion_type,
        'num_classes': num_classes,
        'num_epochs': num_epochs,
        'best_accuracies': [acc.item() for acc in best_acc],
        'mean_accuracy': torch.mean(best_acc).item(),
        'std_accuracy': torch.std(best_acc).item(),
        'final_fusion_weights': [fw[-1] for fw in fusion_weights_list],
        'mean_fusion_weight': np.mean([fw[-1] for fw in fusion_weights_list]),
    }
    
    json_path = f'./log_data_adaptive_fusion/{dateStr}{model_name}summary.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f'Summary saved to: {json_path}')
    
    # Plot training curves
    plot_path = f'./log_data_adaptive_fusion/{dateStr}{model_name}training_curves.png'
    plot_training_curves(acc_list, training_loss_list, test_loss_list, plot_path)
    
    # Plot fusion weight evolution
    if fusion_weights_list[0]:
        fusion_plot_path = f'./log_data_adaptive_fusion/{dateStr}{model_name}fusion_weights.png'
        analyze_fusion_weights(fusion_weights_list[0], fusion_plot_path)
    
    # Print final results
    print('\n' + '='*60)
    print('FINAL RESULTS')
    print('='*60)
    print(f'Fusion Type: {args.fusion_type}')
    print(f'Average Accuracy: {torch.mean(best_acc):.3f}%')
    print(f'Std Accuracy: {torch.std(best_acc):.3f}%')
    print(f'Final Fusion Weight (α): {np.mean([fw[-1] for fw in fusion_weights_list]):.4f}')
    print(f'  (α=1.0 means pure spatial, α=0.0 means pure temporal, α=0.5 means equal)')
    print('='*60)


import os
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io
import torch

class Logger:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        self.history = {
            'epoch': [], 
            'train_loss': [], 
            'train_acc': [],
            'val_loss': [], 
            'val_acc': [],
            'lr': []
        }

    def log(self, epoch, train_loss, train_acc, val_loss, val_acc, lr):
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)
        self.history['lr'].append(lr)
        
        df = pd.DataFrame(self.history)
        df.to_csv(os.path.join(self.output_dir, "training_log.csv"), index=False)

    def save_plots(self,final_test_acc=None):
        plt.figure(figsize=(10, 5))
        plt.plot(self.history['epoch'], self.history['train_acc'], label='Train Acc')
        plt.plot(self.history['epoch'], self.history['val_acc'], label='Val Acc')
        plt.title('Accuracy Curve')
        if final_test_acc is not None:
            plt.axhline(y=final_test_acc, color='r', linestyle='--', alpha=0.6, label=f'Test Acc: {final_test_acc:.2f}%')
            mid_x = self.history['epoch'][len(self.history['epoch']) // 2]
            plt.text(mid_x, final_test_acc + 1.0, f"Test: {final_test_acc:.2f}%", 
                     color='red', fontweight='bold', ha='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
            
            plt.title(f'Accuracy Curve (Best Test: {final_test_acc:.2f}%)')
        else:
            plt.title('Accuracy Curve')        
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, "accuracy_plot.png"))
        plt.close()
        plt.figure(figsize=(10, 5))
        plt.plot(self.history['epoch'], self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['epoch'], self.history['val_loss'], label='Val Loss')
        plt.title('Loss Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, "loss_plot.png"))
        plt.close()

    def save_mat(self):
        scipy.io.savemat(os.path.join(self.output_dir, "metrics.mat"), self.history)
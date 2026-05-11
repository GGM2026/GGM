import os
import torch
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import Subset
class Evaluator:
    def __init__(self, model, test_loader, device, output_dir):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.output_dir = output_dir
        # Get class names from the dataset object
        self.class_names = self.test_loader.dataset.dataset.classes if isinstance(self.test_loader.dataset, Subset) else self.test_loader.dataset.classes

    def plot_confusion_matrix(self, cm, file_path):
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title('Confusion Matrix', fontsize=15)
        plt.savefig(file_path)
        plt.show()

    def evaluate(self, model_path):
        print(f"\n--- Starting Final Evaluation ---")
        # Load model state dictionary with map_location
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device(self.device)))
        #self.model.to(self.device) # check this later if needed or not
        self.model.eval()

        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="Testing"):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # --- Calculate Metrics ---
        report_dict = classification_report(all_labels, all_preds,
                                            target_names=self.class_names, output_dict=True)
        cm = confusion_matrix(all_labels, all_preds)

        # Calculate class-wise accuracy from the confusion matrix
        class_accuracies = cm.diagonal() / cm.sum(axis=1)
        for i, class_name in enumerate(self.class_names):
            report_dict[class_name]['accuracy'] = class_accuracies[i]

        # Add overall accuracy to the report
        report_dict['overall_accuracy'] = accuracy_score(all_labels, all_preds)

        # --- Save & Display Results ---
        print(f"\nOverall Test Accuracy: {report_dict['overall_accuracy'] * 100:.2f}%")

        # Display results in a clean table using pandas
        report_df = pd.DataFrame(report_dict).transpose()
        print("\nClassification Report:")
        print(report_df.to_string())

        # Save report to JSON
        report_path = os.path.join(self.output_dir, "classification_report.json")
        report_df.to_json(report_path, indent=4)
        print(f"\nFull report saved to {report_path}")

        # Plot and save confusion matrix
        cm_path = os.path.join(self.output_dir, "confusion_matrix.png")
        self.plot_confusion_matrix(cm, cm_path)
        print(f"Confusion matrix plot saved to {cm_path}")

        return report_dict

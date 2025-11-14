import os
from keras.models import load_model
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report
import keras
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Create directory for saving images
os.makedirs('evaluation_images', exist_ok=True)

(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Save sample images from the dataset
def save_sample_images(X, y, class_names, num_samples=10, filename='evaluation_images/sample_images.png'):
    plt.figure(figsize=(12, 6))
    for i in range(min(num_samples, len(X))):
        plt.subplot(2, 5, i + 1)
        plt.imshow(X[i])
        plt.title(class_names[int(y[i])])
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()  # Close the figure to free memory
    print(f"Sample images saved to {filename}")

print("Saving sample images from CIFAR-10 dataset...")
save_sample_images(X_test, y_test, class_names)

# List of models to evaluate
models = [
    ("./initial_model.keras", "Baseline CNN"),
    ("./model_with_more_layers.keras", "CNN with More Layers"),
    ("./transfer-learning1.keras", "Transfer Learning (MobileNetV3Large)"),
    ("./transfer-learning2-with-partial-retrain-of-base-model.keras", "Transfer Learning with Fine-tuning"),
]

print(f"\nEvaluating {len(models)} models:\n")

# Prepare report content
report_content = f"# Model Evaluation Report\n\n"
report_content += f"## Dataset: CIFAR-10\n\n"
report_content += f"### Sample Images from CIFAR-10 Dataset\n\n"
report_content += f"![Sample Images](evaluation_images/sample_images.png)\n\n"

# Evaluate each model with comprehensive metrics
for model_path, model_name in models:
    try:
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"Model path: {model_path}")
        print('='*60)

        # Load the model
        model = load_model(model_path)

        # Get predictions
        y_pred_proba = model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)

        # Calculate metrics
        est_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

        # Calculate additional metrics
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        print(f"Test Accuracy:  {test_acc:.4f}")
        print(f"Precision:      {precision:.4f}")
        print(f"Recall:         {recall:.4f}")
        print(f"F1-Score:       {f1:.4f}")

        # Classification report
        class_report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
        print(f"\nClassification Report:")
        print(class_report)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)

        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()

        # Save confusion matrix
        cm_filename = f'evaluation_images/confusion_matrix_{model_name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")}.png'
        plt.savefig(cm_filename)
        plt.close()  # Close the figure to free memory
        print(f"Confusion matrix saved to {cm_filename}")

        print(f"\nConfusion Matrix shape: {cm.shape}")

        # Add model results to report
        report_content += f"## {model_name}\n\n"
        report_content += f"- **Model Path**: {model_path}\n"
        report_content += f"- **Test Accuracy**: {test_acc:.4f}\n"
        report_content += f"- **Precision**: {precision:.4f}\n"
        report_content += f"- **Recall**: {recall:.4f}\n"
        report_content += f"- **F1-Score**: {f1:.4f}\n\n"
        report_content += f"### Confusion Matrix\n\n"
        report_content += f"![Confusion Matrix for {model_name}]({cm_filename})\n\n"
        report_content += f"### Classification Report\n```\n{class_report}\n```\n\n"

    except Exception as e:
        error_msg = f"Error evaluating {model_name} ({model_path}): {str(e)}"
        print(error_msg)
        report_content += f"## {model_name}\n\n"
        report_content += f"- **Error**: {error_msg}\n\n"
        continue

print(f"\nModel evaluation completed.")

# Write the report to file
with open('models_evaluation_report.md', 'w') as f:
    f.write(report_content)

print("Report saved to models_evaluation_report.md")

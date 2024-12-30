import os
import warnings
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from PIL import Image

matplotlib.use('Agg')
warnings.filterwarnings("ignore")

DATASET_PATH = './cifar10/'
def load_cifar10_from_directory(base_path):
    def load_images_from_folder(folder):
        images = []
        labels = []
        label_names = os.listdir(folder)
        for label_idx, label_name in enumerate(label_names):
            label_path = os.path.join(folder, label_name)
            for image_file in os.listdir(label_path):
                image_path = os.path.join(label_path, image_file)
                image = Image.open(image_path).resize((32, 32))
                images.append(np.array(image))
                labels.append(label_idx)
        return np.array(images), np.array(labels)

    train_path = os.path.join(base_path, 'train')
    test_path = os.path.join(base_path, 'test')

    x_train, y_train = load_images_from_folder(train_path)
    x_test, y_test = load_images_from_folder(test_path)

    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = load_cifar10_from_directory(DATASET_PATH)

x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

base_model = GradientBoostingClassifier(random_state=42)

grid_search = GridSearchCV(base_model, param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search.fit(x_train, y_train)

best_params = grid_search.best_params_
print(f"最佳参数: {best_params}")

best_model = GradientBoostingClassifier(
    n_estimators=best_params['n_estimators'],
    learning_rate=best_params['learning_rate'],
    max_depth=best_params['max_depth'],
    random_state=42
)
best_model.fit(x_train, y_train)

y_pred_optimized = best_model.predict(x_test)

train_loss = best_model.train_score_

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_loss) + 1), train_loss, marker='o', label='Training Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Optimized Gradient Boosting Loss Curve')
plt.legend()
plt.grid()
plt.savefig("imgs/op_loss.png", dpi=300)

acc_optimized = accuracy_score(y_test, y_pred_optimized)
report_optimized = classification_report(y_test, y_pred_optimized, zero_division=0)

print("Optimized Gradient Boosting")
print(f"Accuracy: {acc_optimized}")
print(report_optimized)

conf_matrix_optimized = confusion_matrix(y_test, y_pred_optimized)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_optimized, annot=True, fmt="d", cmap="Blues")
plt.title(f"Optimized Confusion Matrix - Gradient Boosting")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("imgs/best_cm.png", dpi=300)

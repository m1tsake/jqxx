import os
import warnings

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, adjusted_rand_score
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

models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    "KMeans": KMeans(n_clusters=10, random_state=42)
}

results = {}

for name, model in models.items():
    if name == "KMeans":
        model.fit(x_train)
        y_pred = model.predict(x_test)
        ari = adjusted_rand_score(y_test, y_pred)
        acc = None
        report = None
        conf_matrix = None
        print(f"Model: {name}")
        print(f"Adjusted Rand Index: {ari}")
    else:
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, zero_division=0)
        conf_matrix = confusion_matrix(y_test, y_pred)
        print(f"Model: {name}")
        print(f"Accuracy: {acc}")
        print(report)

    results[name] = {
        "model": model,
        "accuracy": acc,
        "report": report,
        "conf_matrix": conf_matrix,
        "ari": ari if name == "KMeans" else None
    }

for name, res in results.items():
    if res["conf_matrix"] is not None:
        plt.figure(figsize=(10, 8))
        sns.heatmap(res["conf_matrix"], annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig(f"imgs/{name}_cm.png", dpi=300)

if "GradientBoosting" in results:
    model = results["GradientBoosting"]["model"]
    train_loss = model.train_score_

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_loss) + 1), train_loss, marker='o', label='Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Gradient Boosting Loss Curve')
    plt.legend()
    plt.grid()
    plt.savefig("imgs/loss.png", dpi=300)

best_model_name = max(results, key=lambda k: results[k]["accuracy"] if results[k]["accuracy"] is not None else 0)
print(f"Best model is: {best_model_name}")
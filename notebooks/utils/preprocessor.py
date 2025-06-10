from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np

def show_sample_images(directory, label, num_images=5):
    path = os.path.join(directory, label)
    images = os.listdir(path)[:num_images]
    plt.figure(figsize=(15, 3))
    for i, img_name in enumerate(images):
        img_path = os.path.join(path, img_name)
        img = Image.open(img_path)
        plt.subplot(1, num_images, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(label)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_augmentation(generator, class_label, num_images=5):
    """
    Displays augmented images for a specific class label.
    """
    batch = next(generator)
    images, labels = batch
    indices = np.where(labels == class_label)[0][:num_images]

    plt.figure(figsize=(15, 3))
    for i, idx in enumerate(indices):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[idx].squeeze(), cmap='gray')
        plt.title(f'Label: {int(labels[idx])}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
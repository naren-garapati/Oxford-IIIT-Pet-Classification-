# 🐶 Pet Breed Classification with CNN

This project demonstrates how to build and train a Convolutional Neural Network (CNN) using PyTorch to classify pet images into **37 different breeds**. It includes a custom architecture, a PyTorch `Dataset` wrapper, and a full training pipeline.

---

## 📁 Dataset Assumptions

- Images are stored on disk and accessible via file paths.
- Each image has a corresponding label (integer from 0 to 36).
- Images are in RGB format; grayscale images are converted during preprocessing.

---

## 🧠 Model Architecture: `PetCNN`

The CNN is designed for input images of shape **224×224×3** and outputs probabilities across 37 classes.

**Structure:**

- **Conv Layers**:
  - `Conv2d(3, 32)` → ReLU → MaxPool
  - `Conv2d(32, 64)` → ReLU → MaxPool
  - `Conv2d(64, 128)` → ReLU → MaxPool
  - `Conv2d(128, 256)` → ReLU → MaxPool
- **Flatten**
- **Fully Connected**:
  - `Linear(256×14×14 → 512)` → Dropout → ReLU
  - `Linear(512 → 256)` → Dropout → ReLU
  - `Linear(256 → 37)` (class logits)

Includes `BatchNorm` after each convolution and dropout (p=0.3) before dense layers.

---

## 🧾 Custom Dataset Class: `PetDataset`

A PyTorch `Dataset` wrapper to handle:
- Image loading and label assignment
- RGB conversion for grayscale images
- Custom image transformations (resize, normalization)

**Essential methods**:
```python
def __len__(self):
    return len(self.image_paths)

def __getitem__(self, idx):
    image = Image.open(self.image_paths[idx]).convert('RGB')
    label = self.labels[idx]
    if self.transform:
        image = self.transform(image)
    return image, label

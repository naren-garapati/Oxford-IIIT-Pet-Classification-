# 🐶 Pet Breed Classification using PyTorch CNN

This project aims to classify pet images into one of 37 distinct breeds using a custom-built Convolutional Neural Network (CNN) implemented in PyTorch. The notebook demonstrates the complete deep learning workflow: model definition, dataset creation, training, and evaluation.

---

## 📌 Project Overview

Image classification, particularly for fine-grained tasks like breed identification, is a common challenge in computer vision. In this project:

- A deep CNN is built from scratch (without transfer learning) to identify 37 cat/dog breeds.
- The model is trained using RGB images resized to 224×224 pixels.
- A custom PyTorch `Dataset` class handles image loading and preprocessing.
- Key deep learning techniques such as Batch Normalization, Dropout, and ReLU activations are used to improve performance.
- Two files, one utilizing PyTorch and the other Tensorflow

---

## 🧠 Model Architecture: `PetCNN`

The model processes an input image through four convolutional blocks followed by three fully connected layers. Here's the architectural summary:

**Input**: 224×224×3 (RGB image)  
**Output**: 37 logits representing breed probabilities

**Layers:**
1. **Conv Block 1**: 3→32 filters → BatchNorm → ReLU → MaxPool  
2. **Conv Block 2**: 32→64 filters → BatchNorm → ReLU → MaxPool  
3. **Conv Block 3**: 64→128 filters → BatchNorm → ReLU → MaxPool  
4. **Conv Block 4**: 128→256 filters → BatchNorm → ReLU → MaxPool  
5. **Flatten**  
6. **FC1**: Linear(256×14×14 → 512) → Dropout → ReLU  
7. **FC2**: Linear(512 → 256) → Dropout → ReLU  
8. **FC3**: Linear(256 → 37)

---

## 🗃️ Data Handling

### Dataset Format
- Images are organized by file paths.
- Each image is paired with an integer label (0 to 36).
- All images are converted to RGB if needed.

### Custom Dataset Class: `PetDataset`

Implements PyTorch’s `Dataset` interface:

```python
def __len__(self):
    return len(self.image_paths)

def __getitem__(self, idx):
    image = Image.open(self.image_paths[idx]).convert('RGB')
    label = self.labels[idx]
    if self.transform:
        image = self.transform(image)
    return image, label




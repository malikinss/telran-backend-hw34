# HW34: Synthetic Shape Dataset & YOLOv8 Custom Model

## Task Definition

The goal of this project is to create a **synthetic dataset** of simple geometric shapes (circles and squares), generate corresponding **YOLOv8 labels**, train a **custom object detection model**, and test it on images containing a combination of shapes.

Specifically, the project involves:

- Generating **30 circle images** and **30 square images** for training
- Creating matching **YOLO labels** for each shape
- Generating **3 circles** and **3 squares** for validation with corresponding labels
- Updating the `names` field in `data.yaml` for YOLO compatibility
- Training a **YOLOv8 model** (`best.pt`) on the generated dataset
- Testing the trained model on external images containing a mix of circles and squares
- Implementing the dataset generation and label creation in a **DRY (Don't Repeat Yourself)** manner

This assignment emphasizes automation, modularity, and efficient dataset preparation for computer vision tasks.

---

## 📝 Description

This project provides a complete workflow for **synthetic image generation, labeling, training, and inference**:

1. **Synthetic Dataset Generation**
    - Images are created programmatically using OpenCV.
    - Random positions and sizes are assigned to each shape.
    - Dataset is split into `train` and `val` sets.
    - Labels are generated in **YOLO format** with normalized coordinates.

2. **Data YAML Configuration**
    - The `data.yaml` file contains paths for training and validation images.
    - It also defines `names` mapping class indices to shape names (`circle`, `square`).

3. **Model Training**
    - YOLOv8m is trained on the synthetic dataset using configurable epochs, batch size, and image dimensions.
    - Training automatically produces `best.pt` weights in a designated experiment folder.

4. **Inference & Visualization**
    - The trained model can be applied to new images.
    - Predicted bounding boxes and class labels are visualized using Matplotlib.

---

## 🎯 Purpose

The project demonstrates:

- How to create **custom datasets** for deep learning without manual annotation
- Using **YOLOv8** for object detection tasks
- Automating **label generation** in YOLO format
- Designing **modular, reusable Python code** for dataset generation and training
- Understanding **train/validation splits** and **experiment reproducibility**

This workflow is widely applicable to **computer vision tasks**, prototyping, and learning object detection pipelines.

---

## 🔍 How It Works

### 1. Dataset Generation

- Random parameters are generated for each shape:
    - Circles: `(x_center, y_center, radius)`
    - Squares: `(x_top_left, y_top_left, side_length)`
- Images are saved in `datasets/images/train` or `datasets/images/val`
- Labels are normalized and stored in `datasets/labels/train` or `datasets/labels/val`

All paths are automatically managed using `PathBuilder` functions, avoiding hard-coded file paths.

### 2. Data YAML

The `data.yaml` file contains:

```yaml
path: datasets
train: images/train
val: images/val
names:
    0: circle
    1: square
```

### 3. Training

- YOLOv8m model is trained using the Ultralytics library.
- Training configuration is flexible:
    - `epochs`: Number of epochs (default 20)
    - `img_size`: Input image size (default 256)
    - `batch`: Batch size (default 2)
    - `experiment_name`: Folder name for saving weights (default `circle_exp`)

### 4. Inference

- Load the trained model:

```python
from ultralytics import YOLO

model = YOLO("runs/detect/circle_exp/weights/best.pt")
```

- Run inference on a test image:

```python
results = model("test_img.jpg")
```

- Visualize predictions:

```python
annotated = results[0].plot()
import matplotlib.pyplot as plt

plt.imshow(annotated)
plt.axis("off")
plt.show()
```

---

## 📦 Usage

```python
from src.dataset_generation import create_dataset
from src.generate_data_yaml import generate_data_yaml
from src.training import train_model

# Generate dataset
create_dataset()

# Generate YOLO data.yaml
generate_data_yaml()

# Train YOLOv8 model
train_model(epochs=20, img_size=256, batch=2, experiment_name="circle_exp")
```

After training, `best.pt` can be used for inference on new images.

---

## 🧪 Running Tests

Since the dataset is synthetic, testing consists of:

1. Verifying image and label generation correctness
2. Ensuring YOLO model detects all shapes in validation images
3. Visual inspection of bounding boxes and labels

Optional: Write unit tests for helper functions (path building, label normalization, image generation).

---

## ✅ Dependencies

- Python 3.10+
- OpenCV (`cv2`)
- NumPy (`numpy`)
- Ultralytics YOLOv8 (`ultralytics`)
- Matplotlib (`matplotlib`)
- PyYAML (`yaml`)

---

## 🗂 Project Structure

```
.
├── datasets/
│   ├── images/train
│   ├── images/val
│   ├── labels/train
│   ├── labels/val
|   └── data.yaml
├── src/
│   ├── dataset_generation.py
│   ├── training.py
│   ├── generate_data_yaml.py
│   └── utils/
│       ├── common_constants.py
│       ├── common_type_aliases.py
│       ├── files_generation.py
│       ├── image_generation.py
│       ├── label_generation.py
│       └── training_structure.py
├── circle_test.ipynb
├── main.py
├── README.md
└── datasets/data.yaml
```

---

## 📊 Project Status

**Status:** Completed ✅

- Synthetic dataset generation is automated
- YOLOv8 training and inference pipeline works end-to-end
- Labels are YOLO-compatible
- Modular, reusable code structure

---

## 📄 License

MIT License

---

## 🧮 Conclusion

This project provides a full **end-to-end workflow** for creating synthetic datasets, training object detection models, and performing inference.

It highlights **automation, reproducibility, and modular Python design** while allowing experimentation with **computer vision models**.

---

Made with ❤️ and `Python` by **Sam-Shepsl Malikin** 🎓  
© 2026 All rights reserved.

## Detection of Facial Deepfake images using CNN Architecture

- Built a User Friendly Facial DeepFake detection system with DenseNet121 architecture.<br>
- Achieved a deepfake detection accuracy of 95.16%.<br>

## Project Explanation :

- Model Architecture: DenseNet121, a pre-trained CNN model, is used as the core architecture for classification.

- DenseNet121: A densely connected convolutional network where each layer is connected to every other layer, ensuring efficient feature reuse.
- Pre-trained on: ImageNet dataset, enabling transfer learning for deepfake detection.<br>

- we have used a robust dataset consisting of 140k facial images consisting of both real and fake images,the model has been trained on 70% of the data,15% for validation and 15% for testing.
- In datapreprocessing Face alignment and resizing is done to ensure uniform input.
- Image normalization and augmentation for better model generalization.

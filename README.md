# Emoji Recognizer using Convolutional Neural Network (CNN)

This project implements a Convolutional Neural Network (CNN) to recognize emojis from images. The model is implemented using Keras with TensorFlow backend. The dataset is currently small, but the model demonstrates good efficiency.

## Model Architecture:

The CNN model consists of multiple layers:
- **Convolutional Layers:** Two sets of convolutional layers with batch normalization and max pooling to capture spatial features.
- **Flatten Layer:** To flatten the 2D feature maps into a 1D vector.
- **Dense Layers:** Two dense layers with ReLU activation and dropout for learning high-level features.
- **Output Layer:** Dense layer with softmax activation for emoji classification.

## Data Augmentation:
Data augmentation is applied to the training set using the `ImageDataGenerator` from Keras. This helps increase the diversity of the training set by applying random transformations like rotation, horizontal flip, shear, height shift, and zoom.

## Training:

The model is trained using the `fit_generator` method. The training data is loaded from the 'datasets' directory with images resized to (64, 64) and a batch size of 10. The model is trained for 5 epochs with 3 steps per epoch. Adjustments to the number of steps and epochs can be made based on the dataset size.

## Testing:

An example image ('datasets/hand/img30.png') is loaded for testing. The image is preprocessed, and the model predicts the emoji class probabilities. The predicted class is then determined using `np.argmax`. The class indices used during training are printed for reference.

## Usage:

1. Ensure the required libraries are installed (`keras`, `numpy`, `tensorflow`).
2. Organize your dataset in the 'datasets' directory with subdirectories for each emoji class.
3. Run the script to train the model and make predictions.

Feel free to expand the dataset for better performance as more diverse data becomes available.

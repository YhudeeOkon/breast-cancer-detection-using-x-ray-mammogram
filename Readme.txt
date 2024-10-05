Requirements

tensorflow
numpy
pandas
cv2 (OpenCV)
matplotlib
seaborn
sklearn

Data Structure

meta.csv: Metadata about the breast cancer cases.
dicom_info.csv: DICOM image paths and descriptions.

Image Preprocessing
The image_processor function resizes and normalizes images:

Resizes the images to a target size of (224, 224)
Normalizes pixel values to a range of [0, 1]

Path Fixing
The paths for the full mammograms, cropped images, and ROI mask images are fixed using a dictionary mapping based on the dicom_info.csv.

Model Architecture
A Convolutional Neural Network (CNN) model is as follows:

Flatten Layer: Flattens the input image for further processing.
Dense Layer (128 units, ReLU activation): First fully connected layer.
Dropout Layer (0.5): Helps reduce overfitting.
Dense Layer (64 units, ReLU activation): Second fully connected layer.
Dropout Layer (0.5): Further regularization.
Output Layer (1 unit, sigmoid activation): Final binary classification (malignant/benign).

Model Training
The mammogram images are split into training, validation, and test sets using an 80-20 train-test split, followed by a validation split from the test set.
The labels are one-hot encoded using to_categorical.
The model is trained for a specified number of epochs, and the performance on validation data is tracked.

Evaluation
The model is evaluated using:

Accuracy and loss on validation data.
Confusion matrix to visualize classification performance.
Classification report: Includes precision, recall, F1-score, and support.


Visualization
Image Display: Functions such as display_images() are used to display mammogram images and overlay predictions.
Accuracy and Loss Plot: Training and validation accuracy/loss over epochs are plotted using the plot_accuracy_and_loss() function.
Confusion Matrix: A heatmap of the confusion matrix is plotted to show the classification results.

How to Run
Ensuring all required libraries are installed.
Organizing the dataset as described in the Data Structure section.
Running the Python script in a Jupyter Notebook or Python environment.

Results
Training and Validation Accuracy: The model achieves high accuracy over the epochs.
Confusion Matrix: Visualizes the true positives, true negatives, false positives, and false negatives.
Classification Report: Summarizes precision, recall, and F1-score for each class.
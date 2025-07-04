# Face Recognition System using CNN on LFW Dataset


Project Overview
This project implements a basic face recognition system using a Convolutional Neural Network (CNN) on a subset of the Labeled Faces in the Wild (LFW) dataset.

The goal was to build a system that can identify individuals based on their facial features extracted by a pre-trained CNN model. Due to the constraints of readily available models in standard libraries within this environment, a ResNet50V2 model pre-trained on ImageNet was used as a feature extractor, although dedicated face recognition models like FaceNet or VGGFace are typically more suitable for this task.

The process involved:
- Loading and preparing the LFW dataset.
- Preprocessing the images (resizing and normalization).
- Extracting facial features (embeddings) using the CNN model.
- Implementing a recognition system based on cosine similarity of embeddings.
- Evaluating the system's performance.
Setup and Running the Code
The code is provided in a Google Colab notebook format. To run the code:

1. Open the notebook in Google Colab.
2. Ensure you have a stable internet connection to download necessary libraries and model weights.
3. Run each code cell sequentially.

The notebook automatically handles the installation of required libraries (like TensorFlow, scikit-learn, matplotlib, seaborn).
Data
The LFW dataset is automatically downloaded and loaded using sklearn.datasets.fetch_lfw_people, specifically retrieving individuals with at least 70 images to focus on a smaller, more manageable subset for this demonstration.
Results
The implemented system, using ResNet50V2 as a feature extractor, achieved an overall face recognition accuracy of approximately 52.48% on the selected subset of the LFW dataset.
Visualizations provided in the notebook include:

- Accuracy per Person: A bar chart titled "Face Recognition Accuracy Per Person" showing the recognition accuracy for each individual in the dataset.
  ![Face_Recognition](https://github.com/user-attachments/assets/2e8f7709-2ff9-4862-899b-b8356f9be0ad)

- Prediction Examples: Examples of correct and incorrect face recognition predictions, displaying the original image and the most similar image found by the system.
- Confusion Matrix: A heatmap titled "Confusion Matrix for Face Recognition" illustrating the true vs. predicted identities for each image.
  ![confusion_matrix](https://github.com/user-attachments/assets/7cd01877-91d6-46e4-95e0-74be5e96aeea)

Note: This accuracy is relatively low for a face recognition task, highlighting the importance of using models specifically trained on large face datasets (like FaceNet or VGGFace) for better performance. The current implementation serves as a demonstration of the steps involved using a generally available CNN.
Future Improvements
- Integrate a dedicated face recognition model (FaceNet, VGGFace).
- Implement pairwise verification as the standard evaluation metric for LFW.
- Explore fine-tuning the chosen model on a face dataset.

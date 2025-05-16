Overview
This project focuses on developing a robust skin disease classification model using deep learning techniques. The model is trained on a large dataset containing 21,720 images categorized into 10 different skin disease classes. To address challenges like class imbalance and improve model generalization, advanced techniques such as class weighting, CutMix, and MixUp data augmentation are employed.

The architecture is based on the MobileNetV2 pretrained model, optimized to work efficiently with high accuracy while being lightweight enough for potential deployment on mobile devices or low-resource environments.

Dataset
Total images: 21,720

Number of classes: 10

The dataset contains labeled images of various common skin diseases.

The dataset was preprocessed and split into training, validation, and test sets.

Features
MobileNetV2 Backbone: Leveraging a lightweight and efficient pretrained convolutional neural network that is well-suited for image classification tasks.

Class Imbalance Handling: Used class weights during training to mitigate bias toward majority classes and improve performance on underrepresented classes.

Data Augmentation: Applied CutMix and MixUp augmentation techniques to increase data diversity and help the model generalize better on unseen data.

Transfer Learning: Fine-tuned a pretrained model instead of training from scratch to leverage learned features and reduce training time.

Development Environment: Entire workflow implemented using Jupyter Notebook for interactive development and visualization.

Methodology
Data Preparation:

Images resized and normalized.

Dataset split into training, validation, and test sets.

Class weights computed based on class frequency.

Model Training:

Initialized MobileNetV2 with pretrained ImageNet weights.

Used class weights in loss function to balance the classes.

Applied CutMix and MixUp augmentation in the data pipeline.

Optimized using Adam optimizer with an appropriate learning rate.

Early stopping and model checkpointing implemented to avoid overfitting.

Evaluation:

Model evaluated using metrics such as accuracy, precision, recall, and F1-score.

Confusion matrix generated to analyze class-wise performance.

Libraries and Tools Used
The following Python libraries were used throughout the project:

tensorflow (or tensorflow.keras) — for building and training the deep learning model.

numpy — for numerical operations and array manipulations.

pandas — for handling datasets and CSV file operations.

matplotlib & seaborn — for plotting training curves, confusion matrix, and other visualizations.

scikit-learn (sklearn) — for metrics like accuracy, precision, recall, F1-score, and train-test split utilities.

opencv-python (cv2) — for image processing tasks like reading and resizing images.

imgaug or albumentations (optional) — for advanced image augmentations including CutMix and MixUp implementations.

jupyter — for interactive notebook environment to run, debug, and visualize the training and evaluation steps.

os and glob — for file handling and directory traversal.

tensorflow_addons (optional) — for some advanced loss functions or optimizers if used.

Note: Exact libraries and versions can be found in the requirements.txt file.

How to Run
Clone the repository:

bash
Copy
Edit
git clone https://github.com/your-username/skin-disease-prediction.git
cd skin-disease-prediction
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the Jupyter Notebook:

bash
Copy
Edit
jupyter notebook
Open the notebook file (Skin_Disease_Prediction.ipynb) and run the cells step-by-step.

Alternatively, to train the model via script:

bash
Copy
Edit
python train.py --dataset_path ./data --epochs 50 --batch_size 32
Test on new images:

bash
Copy
Edit
python predict.py --image_path ./test_images/sample.jpg
Results
The model demonstrates strong performance in classifying skin diseases.

Handling class imbalance and using advanced augmentations improved prediction accuracy across minority classes.

MobileNetV2 provides a good balance between accuracy and computational efficiency.

Future Work
Experiment with other state-of-the-art architectures such as EfficientNet and ResNet variants.

Deploy the model as a mobile app or web service for real-time skin disease detection.

Enhancing the severity prediction logic to more accurate levels with use of more diverse data

Multi Lingual Compatibility

Incorporate more diverse datasets and real-world clinical images for enhanced robustness.

Explore explainable AI methods to provide interpretability for model predictions.

References
MobileNetV2 Paper

CutMix: https://arxiv.org/abs/1905.04899

MixUp: https://arxiv.org/abs/1710.09412


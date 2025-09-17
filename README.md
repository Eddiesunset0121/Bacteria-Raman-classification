# Project: Raman Classification of 30 Bacteria Species

<img width="2325" height="2539" alt="Simple Use case diagram (1)" src="https://github.com/user-attachments/assets/e83658c7-26d9-4298-8a28-4e9a02abbae1" />


## Project Structure
The repository is organized as follows:

/data/raw: Contains the original, raw datasets.

/notebooks: Contains the Jupyter Notebook with the main analysis.

/src: Contains helper functions and scripts.

## Project Outline
Inspired by the paper "Rapid identification of pathogenic bacteria using Raman spectroscopy and deep learning". https://www.nature.com/articles/s41467-019-12898-9

This project develops a complete machine learning pipeline to classify 30 different bacterial species using Raman spectroscopy. The core of the project is a sophisticated One-Dimensional Convolutional Neural Network (1D CNN), which is first trained from scratch and then adapted for real-world clinical use cases through a two-stage transfer learning process.

## Project Detail
🌟 Key Skills & Tools

TensorFlow & Keras for deep learning

Scikit-learn for baseline modeling and preprocessing

Pandas & Numpy for data manipulation

Matplotlib & Seaborn for data visualization

🌿 Analysis & Key Findings

A comprehensive workflow including EDA, baseline modeling, iterative 1D CNN development, a two-stage transfer learning process, and model interpretation with Grad-CAM.

🌿 Foundational Analysis & Preprocessing:

The provided spectral data was confirmed to be of high quality, with baseline correction and cosmic ray removal already performed.

EDA revealed that the training data was ordered by class, which could cause learning instability. This finding underscored the necessity of 

shuffling the data during the training process to ensure the model converges smoothly.

🌿 Model Development & Selection:

A PCA + SVC model was first developed to establish a baseline, achieving an accuracy of 41.4%. This demonstrated the complexity of the 30-class problem and justified a deep learning approach.


A 1D CNN was then built and iteratively improved. The most critical enhancement was the two-stage transfer learning process. Architectural changes alone were insufficient, but adapting the model to new data distributions was the key to success.


🌿 Final Model Performance:

After the first transfer learning stage to correct for "equipment drift," the model's accuracy on the test set jumped from 53.7% to 84.3%.

The final model, adapted for a 5-class clinical dataset, demonstrates outstanding predictive power, achieving an overall accuracy of 97.6% on unseen clinical test data.


The confusion matrix for the final model shows excellent performance, with diagonal accuracies ranging from 96% to 99%, making it a highly reliable tool for clinical classification.



💡 Clinical Application

Rapid Identification: Offers a method for identifying pathogenic bacteria in minutes, a significant speed improvement over traditional multi-day culturing methods.

Point-of-Care Diagnostics: With portable Raman spectrometers, this model could be deployed in clinics for on-the-spot infection identification, enabling faster treatment decisions.

Robust & Adaptable: The model is specifically designed to be robust against real-world challenges. The transfer learning stages have proven its ability to adapt to variations from equipment drift and the increased noise inherent in clinical samples.

# DecisionTree_ML
This is part of my programming assignments for the course CSCE-633 Machine Learning Course in Texas A&M University.

The assignments consisted of implementing various classification algorithms such as k-means, SVM, Decision Trees on different dataset. 

PCA was applied on **Yale Face Dataset** to compute top 10 eigenfaces which was later used for face recognition by applying k-means classifier. Convolution Neural Netowrk was also used on the same dataset to compare performance with k-means classifier. Data-Augmentation was applied to improve performance(Accuracy).

# Project - Emotional Recognition of Human Faces
As part of Final Project for this course, I worked on project that can detect human emotions with the images of peoples facial expression
and offer advice and recommendations that can help them quell their negative emotions if detected. A deep neural network ensemble model was developed to predict facial emotions. Based on these predictions, the designed system suggests the user way to improve their mood. The dataset we used is **FER2013** human faces dataset with seven emotion labels. Firstly the data was cleaned by cropping out the faces from the whole image. The next step was **Data Augmentation** as this increases the training set. We have done experiments using VGGnet, Resnet50, mini-Xception, Inception-V3, Traditional CNN and an Ensemble model.



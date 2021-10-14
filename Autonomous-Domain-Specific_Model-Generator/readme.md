# Autonomous Domain-specific Model Generator

## Introduction
<div style="text-align: justify">
The main challenging issue to utilize deep learning for various tasks such as content's popularity prediction is to efficiently choose the best-suited deep learning architecture among the various types of deep learning. Currently, domain-specific deep learning models are constructed manually by human experts, where the model construction process is a time-consuming and error-prone process. This is because the human experts need to find out the appropriate neural architectures, training procedures, regularization methods, and hyperparameters of all of these components in order to make their networks do what they are supposed to do with sufficient performance. This process has to be repeated for every application. Because of this, we are developing the opensource platform to autonomously search the appropriate deep learning architecture for problem-specific tasks (i.e., find the appropriate model among feedforward neural networks, recurrent neural networks, etc. )as well as autonomously do the hyperparameters optimization (i.e., find the number of appropriate hidden layers, number of neurons, etc. ).
</div>

## The progress of Autonomous Domain-specific Model Generator 
<div style="text-align: justify">
We implemented the “Autonomous Domain-specific Model Generator” (ADSMG) by using Tensorflow with Keras. Currently, ADSMG is underdevelopement and we implemented the following basic components for the ADSMG as shown in Figure 1. The implemented components are as follows: 1) Data Store, 2) Preprocessing Module, 3) Model Architecture Dictionary, Model Creation Module, Temporary Model Store, Model Training Module, and Model store. Currently, the ADSMG can assess via Jupyter Notebook. In the current version, the Data Store is to store the open dataset from the various sources such as “MovieLens” dataset. The preprocessing module cleans the raw data to get the trainable data for the best-suited model searching and selection process. The Model Architecture Dictionary keeps the general deep learning framework (i.e., Convolutional Neural Network) and rules to construct deep learning models. Then, the Model Creation Module generates the potential deep learning model by using various types of model searching and construction algorithms (i.e., random search). Then, the Model Creation Module store all of the constructed models and the performance of the models in the Tmp Model Store. Next, the Model Training Module chooses the best-suited model and train that model with the large dataset. Finally, the best-suited model is stored at the Model Store for future use.
</div>
</br>
![system_model](https://github.com/kyithar/Autonomous-Domain-specific-Model-Generator-/blob/master/md_figs/system_model.jpg)

Fig. 1 Basic components of Autonomous Domain-specific Model Generator

## Requirements

Tensorflow

Keras

Pandas

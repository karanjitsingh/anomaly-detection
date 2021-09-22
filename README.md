# anomaly-detection
## Proposal: Anomaly Detection in Time Series using AutoEncoders

### Abstract

With almost every problem that involves similarity detection or categorical classification, the subject of classification in the collected data must be statistically significant for a model to perform relatively well. Such classification or similarity detection becomes much harder when the statistical presence of information of interest in the dataset is much lesser. For such anomaly detection, we propose a different paradigm than typical classification and clustering techniques. Instead of explicitly classifying the anomalies, we focus on modeling the predominant normality of the dataset and hence endeavor to detect deviations in the form of anomalies.

### Introduction and Motivation

In typical classifications and classification models, the specific classes of interest to be labeled usually predominate the data. In contrast, Anomaly detection techniques focus on labelling and classifying datasets where the occurrence of subject is statistically rare. In certain scenarios where such occurrences in datasets are extremely rare, even traditional anomaly detection techniques fail to detect occurrences accurately. Additionally, for such scenarios data labelling can prove even more difficult.

This project focuses on anomaly detection in time series problems. We make two assumptions about the dataset. First, we assume that most of the given data is normal and only a very small percentage is abnormal. Second, we anticipate abnormal data is statistically various from normal. According to these two assumptions, data groups of similar instances which appear frequently are assumed to be normal, while infrequently instances which considerably various from the majority of the instances are regarded anomalous. For our experimentation we employ a different approach to anomaly detection. Instead of following in the path of traditional techniques and modeling the infrequent anomalies of interest in the dataset, we focus on models that train on the predominant &quot;normal&quot; scenarios of dataset to detect deviations in the datasets in hopes to detect the anomalies.

One such problems in anomaly detection of time series is Daphnet Freezing of Gait (FOG) [1]. The FOG dataset is devised to benchmark automatic methods to recognize gait freeze from wearable acceleration sensors placed on legs and hips. The dataset was recorded in the lab with emphasis on generating many freeze events. Users performed three kinds of tasks: straight line walking, walking with numerous turns, and finally a more realistic activity of daily living (ADL) task, where users went into different rooms while fetching coffee, opening doors, etc.

For this project, we mainly work with proven time series ensembles such as neural networks including LSTM and convolutional techniques as well as support vector machines. We experiment with both supervised and unsupervised methodologies, while focusing on unsupervised techniques as they best fit the problem description. The goal of the project is to train these models on time series datasets in order to detect anomalies and deviations in the time series.

### Existing Work on Anomaly Detection

Anomaly detection is a vast field in machine learning. We surveyed the major techniques for general anomaly detection as well as anomaly detection in time series in the machine learning paradigm. Both categories contain machine learning techniques. The paper _&quot;Machine Learning Techniques for Anomaly Detection&quot;_ [2], overviews most general techniques and serves as our understanding for anomaly detection. The paper, however, only covers possible models from both supervised and unsupervised categories on anomaly detection without generalized experimentation. The most notable models that stand out for our problem set are unsupervised neural networks and One-Class SVMs. As described in the paper, these techniques do not need training data. Both these models are assumed as single class outputs which can be trained on a given dataset whose output can be interpreted as confidence intervals for the normality of the test input.

Another paper _&quot;Anomaly detection in time series&quot;_ [3], discusses techniques that can be used in this project. This paper drills into the nature of time series and discusses methods that can be used to detect anomalies in long streaming time series as well as short / long instances of time series data.

An existing work with neural networks on Daphnet Freezing of Gait, DeepConvLSTM [4] detects anomalies in the said dataset using supervised neural networks for classification of the anomalies. This paper is best fit to serve as our baseline for experiments with the FOG dataset.

### References

[1] Bächlin, M., Plotnik, M., Roggen, D., Maidan, I., Hausdorff, J. M., Giladi, N., &amp; Tröster, G. (2010). Wearable assistant for Parkinson&#39;s disease patients with the freezing of gait symptom. _IEEE transactions on information technology in biomedicine : a publication of the IEEE Engineering in Medicine and Biology Society, 14(2)_, 436–446. https://doi.org/10.1109/TITB.2009.2036165

[2] Omar, S., Ngadi, M., Jebur, H., &amp; Benqdara, S. (2013). Machine Learning Techniques for Anomaly Detection: An Overview_. International Journal of Computer Applications, 79._ http://dx.doi.org/10.5120/13715-1478

[3] Teng, M. (2010). Anomaly detection on time series. _2010 IEEE International Conference on Progress in Informatics and Computing_, _1_, 603–608. https://doi.org/10.1109/PIC.2010.5687485

[4] Ordóñez, F., &amp; Roggen, D. (2016). Deep Convolutional and LSTM Recurrent Neural Networks for Multimodal Wearable Activity Recognition. _Sensors, 16(1)_, 115. https://doi.org/10.3390/s16010115
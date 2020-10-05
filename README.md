# Anomaly_Detection_Resources_4_Beginners
**Curated List of resources for the Anomaly detection task for the beginners. The work is created in the paradigms of academic requirements of University of Udine, to assist the new grad students in this field.**

# What is [Anomaly Detection?](https://en.wikipedia.org/wiki/Anomaly_detection)
**In Simple Terms**: Anomaly detection is defined as the identification of samples which significantly differ from a reference set of normal data, under the assumption that the number of anomalies is much smaller than the number of normal samples.

**In more rigorous terms**: From a rigorous point of view the problem is not well defined, as there is no rigorous formalization of the properties this difference should have in order to be considered significant: the definition of anomaly is often dependent on an arbitrarily-defined threshold.

However,from a practical point of view, anomaly detection is a widespread problem, marking strong presence in the fields of medical imaging, network intrusion detection, defect detection, fault prevention, video surveillance, and many others. In medical imaging an anomaly could be a tumor tissue among several data of healthy patients, in quality-assurance industrial inspection it can be a defective product, in the surveillance videos of a shopping mall it can bethe behavioral pattern of a thief compared to normal
clients, etc. Hence, practically anomalies can be of three types :
* Point Anomaly
* Collective or Group Anomaly
* Contextual or Conditional Anomaly
*The below image shows the definition with an example of all three.*

![Anomalies Types](Images/anomaly_types.PNG)

# Role of Machine Learning(ML) and Deep Learning/Artificial Intelligence (AI) in this field
Modern IT infrastructures are more and more oriented to the acquisition of enormous amounts of data, which cannot be manually analyzed and require proper algorithms to be processed. Hence, ML and AI foray into this field to solve the problems. The topic has many potential application field, as discussed above.Many classical machine learning techniques have been adopted to identify anomalies in data, such as *Bayesian networks, rule-based systems, clustering algorithms, statistical analysis, etc.* One of the most popular approaches relies on `Support Vector Machines` and in particular on their `one-class variant`, in which the standard SVM technique is used to split the feature space in two parts, one with high-density data (the normal class) and the other with outliers.

Also, the initial attempts to use the deep learning for the anomaly detection was limited to the use of deep models as the feature extractors. And then later these features were were used for the anomaly detection, using `SVM, decision tree or Isolation forest` methods. These approaches leads to the development of *hybrid deep learning models*, where both lates deep learning and classical machine learning appraoches were combines to solve the anomaly detection problems.

Hence, earlier deep (and ML) anomaly detection methods can eb classified as -
* Novelty Detection
* Outlier Detection

![classification of classical anomaly detection methods](Images/deeep_anom_detect.PNG)

## Novelty Detection

![Novelty Detection](Images/novelity_detect.PNG)

**Novelty detection** is the mechanism by which an intelligent organism/system is able to identify an incoming sensory pattern as being until now unknown. This has huge application in bio-medical fields, manufacturing fields, banks and online transactions, trading etc.

## Outlier Detection

![Outlier Detection](Images/outlier_detect.PNG)

**An outlier** is an observation that diverges from an overall pattern on a sample.

## Difference between Outlier Detection and Novelty Detection

Outlier Detection                       |  Novelty Detection
-------------                           | -------------
The training data contains outliers     |The training data is not polluted
which are defined as observations       |by outliers and we are interested 
that are far from the others.           |in detecting whether a new observation
Outlier detection estimators            |is an outlier. In this context an outlier is
thus try to fit the regions             |also called a novelty
where the training data is              |
the most concentrated,                  |
ignoring the deviant                    |
observations.                           |


## In terms of Learning

In terms of learning Novelty Detection is a **Semi-Supervised Learning**(discussed below), while Outlier detection is **Unsupervised Learning.**(discussed below)

![In terms of learning](Images/terms_of_learning.PNG)

# Anomaly Detection and Deep Learning/Artificial Intelligence/Computer Vision

Despite huge interest of research community and industrialinterest, the topic has been not fully exploited with the modern state-of-the-art deep learning approaches. The classicial computer vision approaches were still in use for the Image anomaly detection for identifying the faulty product on a production line. Hence, Our lab [AVIRES](https://avires.dimi.uniud.it/) is working and developing techniques to tackle these problems.

In the academic literature, the deep learning methods used for solving the anomaly detection task is known as the `"Deep Anomaly Detection`. Deep Anomaly Detection methods can be classified as three types:

* **Fully-Supervised/Supervised Methods**: It is assumed that both normal and anomalous data are available for training, and the problem reduces to a standard classification task. In this case, the main difference between anomaly detection and other classification problems is the *imbalanced nature of the dataset*: anomalies may be available for training, but their amount is by definition much smaller than normal data.

* **Semi-Supervised Methods**: In this case, the assumption is that a labeled training set exists, but it contains only normal data. This is a frequent scenario in real-world applications, where large amounts of normal data can be acquired easily but anomalous ones are extremely rare. The task of the anomaly detection system is thus to learn a “normality” model from the training data, and subsequently classify as anomaly any sample that does not fit the model.

* **Unsupervised Methods**: The unsupervised approach considers the case of unlabeled training sets containing both normal data and anomalies. It consists in estimating the region
where normal data is mostly concentrated, consequently separating the deviant observations. Since, neural networks are not particularly well-suited for clustering tasks, the most common approach in this case is to use the neural network as a feature extractor, and the features are later analyzed with other machine learning tools such as one-class Support Vector Machines.

# Industrial Challenges and Deep Anomaly Methods

![Industrial Challenges](Images/industrial_challenges.PNG)

Since, anomaly detection task has huge industrial application, many industries wants to develop there own specialized methods which suits their requirements of anomaly classification is a complex working domain. The major and prime industrial challenge is "lack of data". Many indsutrial facilities still doesn't have proper infrastrucutre or proper data collection facilities. Even if they have they have huge data imbalance, as they have huge data for the normal product while very few images of the abnormalities. Quality of data is also important and many times industries find it very difficult to arrange that, as they may have hughe piles of messy data.

Apart from the the above mentioned issues, it's also costly to arrange data s maybe entire production dates may get affected. And even if we get the data....**DATA ANNOTATION**: Art of labelling the dataset, is costly(both in terms of time and money).

# Real Life Industial Product Anomalies

With some recent advances in the deep anomaly work and the challenges mentioned above...below you can see how a real life anomalies of industrial products look like-

## 1. Example

<img src="Images/018.png" width="300" height="300"> <img src="Images/005.png" width="300" height="300">

The above images shows a broken beer bottle and a broken pills from an industrial production. Now both of these products are high speed manufacturing products. And we can see that the broken pieces can be small and large. Hence, the challenge will be that our deep model should identify the anomaly at a real life high speed.

## 2. Example

<img src="Images/000.png" width="300" height="300"> <img src="Images/016.png" width="300" height="300">

The above figure shows that how complicated an anomaly can be. You can see that anomalies are localised in a very small area like a raptured capsule or a misprint. The detection of misprint is a very challenging task, where our industrial product is of very high speed production like capsules.

## 3. Example

<img src="Images/006.png" width="300" height="300"> <img src="Images/010.png" width="300" height="300">

This final example in the above images shows texture anomalies. Anomalies found in the carpet production or tile production are surface anoamies or texture anomalies. These are very challenging task to identify as many deep learning methods based on reconstruction error fails to capture these minute details.

# Notes  (How to use this repository)

This repo is made for the beginners in the field of anomaly detection and the resources shared here are very basic. Here is a quick note on how to use this repository

This repo focuses on two things :

* Coding (primarily using Pytorch)
* Some Reading Materials

## Coding
Inside the `Anomaly detection Resources` folder, you can find three folder and two presentations. These are (They all contains links to the repos or codes in the readme section and some actual codes):

* **Advance_Pytorch**: This is a curated list of a huge pytorch work ranging from all field of ML, AI, Computer Vision, DL etc
* **Python Codes Anomaly Detection**: This contains repo links for the basic autoencoders and unsupervised defect segmentation.
* **Pytorch_beginner_resources**: It contains crash course links for the Pytorch, SKLearn anomaly detection module, classical anomaly detection resources.

## Some Reading Materials
Inside the `Anomaly detection Resources` folder you can find two presentations of our lab work (one in english and one in Italian). These are some refereshers and intial research work done in this field by our lab Avires.

# Please feel free for any addition to these resources. Please pull a branch and make a commit request. You can also write to `pankajmishra000@gmail.com`.







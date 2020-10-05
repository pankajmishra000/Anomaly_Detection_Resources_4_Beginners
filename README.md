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

![classification of classical anomaly detection methods](Images/deep_anom_detect.PNG)

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







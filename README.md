# Alzheimer's disease prediction based on continuous feature representation using multi-omics data integration
Alzheimer's disease prediction based on autoencoder and multi-omics data integration

We developed an approach for predicting one of the most common neurological disorders, Alzheimer's disease, using gene expression (GEO ID: GSE33000 and GSE44770) and DNA methylation (GEO ID: GSE80970) datasets.

## Abstract
Alzheimer's disease (AD) is a neurological disease characterized by complex molecular pathways and neural tissue complexity. Investigation into its molecular structure and mechanisms are ongoing, and no therapeutically useful genetic risk factors have been identified. As a result, brain images such as magnetic resonance imaging (MRI) and cognitive testing have been used to diagnose AD. Recently, various independent studies have generated and evaluated large-scale omics data from various brain regions, including the prefrontal cortex. Therefore, strategies for detecting or predicting AD must be developed using these data. In addition, integration of these omics data can be a valuable resource for gaining a more thorough understanding of the disease. This study developed a machine-learning-based approach for predicting AD using DNA-methylation and gene expression datasets. It is one of the challenging tasks to manage these data while building a prediction model since these contain tens of thousands of features and have a high dimensional and low sample size (HDLSS) characteristic. To solve this dilemma, we employed an autoencoder (AE) to generate minimized and continuous feature representation. We used multiple machine-learning approaches to predict AD after receiving the encoded data and calculated the accuracy and area under the curve (AUC). Furthermore, we showed that combining DNA methylation and gene expression data can increase the prediction accuracy. Finally, we compared our method to state-of-the-art technique and found that the proposed methodology outperformed it by improving the accuracy and AUC by 9.5 and 10.6%, respectively.

**For more details, please refer to the [paper](https://doi.org/10.1016/j.chemolab.2022.104536)**


## Flowchart
![flow_diagram](https://user-images.githubusercontent.com/80881943/129854424-7209e13d-558f-41cd-9300-4c40ca473b15.png)



## Specifications
* Python 3.7
* numpy 1.19.4
* pandas 1.1.0
* tensorflow 2.1.0
* keras 2.3.1


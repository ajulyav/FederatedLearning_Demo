# FederatedLearning_Demo

## DEMONSTRATOR 1: PROOF OF CONCEPT OF THE FEDERATED LEARNING PLATFORM

### Introduction and Motivation 

This report focuses on the federated learning aspect as the important direction of multi-hospital cooperation. In particular, the objective is on attemps to start the cross-country  coordination between X, Y and Z towards running the first federated learning experiments in the typical scenario of data variations amongst medical centers. In the end, the successful solution permits secure multi-institutional cooperation, and can be used to enhance/validate models without revealing patient data. Thus, resolving issues with data ownership, privacy, and legality.

In this regard, Federated machine learning (FL) is a promising tool proposing the idea of a machine learning model that is jointly trained by a number of parties (i.e. medical centers, hospitals) while all training data is kept locally and privately. Instead of transferring data directly, computation is done on each site, where model updates are computed and subsequently integrated into a global model using an aggregation mechanism.

### 1. Research Problem

To focus on the feasibility part of building the Federated Learning platform and partnership between sites, the main goal is not proposing a new deep learning solution to tackle a problem. Thus, the key motivation behind choosing the research problem which meets the needs to prove the concept of Federated Learning is based on the public multi-centric dataset and relevance of the task to the medical area. 

Based on this assumption, prostate segmentation was chosen as a research problem due to its importance and available dataset suitable to imitate as closely as possible a real-world scenario. According to Ferlay et al. [1], prostate cancer (CaP) is substantial, ranking among the top five cancers for both incidence and mortality. Globally, prostate cancer is the most commonly diagnosed cancer in men, with approximately 1.6 million incident cases in 2015 [2]. In its turn, the imaging techniques offered by magnetic resonance imaging (MRI) enable the diagnosis and localization of CaP.

In particular, in the framework of the CaP study, T2w sequences are frequently used to manually segment prostate zones. Though even for an experienced clinician, there are a number of challenges that make this task difficult and time-consuming. Foremost, due to physiological variations in tissue intensities, shape, and size, the prostate is prone to significant intersubject variability. Additionally, sequences obtained from various MRI machines increase the variability in the prostate's appearance in T2w imaging.

These challenges are presented to the new level when dealing with the data from multiple medical centers as there could be stronger inter-site heterogeneity due to variations in imaging protocols, endorectal coil usages, or demographics (see Table 2). 

Coming to the assumption that convolutional neural networks (CNNs) have recently made automatic prostatic segmentation possible, effectively learning a robust and accurate segmentation CNN by taking advantage of multiple sources of data could help provide a more generalized model. In this scenario, the Federated Learning platform can be a great tool to facilitate multi-centric research.


#### 1.1 Data: A Multi-site Dataset for Prostate MRI Segmentation

To conduct the Federated Learning experiments in the framework of Demonstrator 1, a part of a public multi-site dataset for prostate MRI segmentation is taken which contains prostate T2-weighted MRI and segmentation masks. Below, there is the table presenting the selected datasets:

Table 1 – Datasets for Federated Learning experiment

|  Dataset |Institution (Client)   |  Number of images | Manufacturer  | Resolution  | Field strength   |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
|  I2CVB [3] - HCRUDB  | X  | 19 // 20 (Training Size: 32, Validation Size: 7)  | Siemens // General Electric (GE)  |0.67-0.79;1.25 // 0.27;3-3.5   |  3.0 Tesla // 1.5 Tesla |
| MSD [4] - RUNMC  | Y  |  32  (Training Size: 26, Validation Size: 6)  | Siemens |  0.6-0.625;3.6-4 | 3 Tesla  |
| NCI-ISBI [5] - BMC  | Z  | 39 (Training Size: 32, Validation Size: 7)  | Philips | 0.4;3  | 1.5 Tesla  |


Among them: 

    • I2CVB – Original Multi-parametric MRI Images of Prostate: http://i2cvb.github.io/ 
    • MSD – Medical Segmentation Decathlon Challenge: http://medicaldecathlon.com/  
    • NCI-ISBI - NCI-ISBI 2013 Challenge – Automated Segmentation of Prostate Structures: https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=21267207#21267207d170e52bc57d4c67b747b57bf88c460f 

The following table illustrates how the three sites differ in appearance:

Table 2 – Differences in appearance 

|   | I2CVB [1] - X  | MSD [2] - Y  |  NCI-ISBI [3] - Z |
| ------------ | ------------ | ------------ | ------------ |
| Sample 1  |  ![](https://github.com/ajulyav/FederatedLearning_Demo/blob/main/imgs/sampl1_1.png?raw=true)|  ![](https://github.com/ajulyav/FederatedLearning_Demo/blob/main/imgs/sampl1_2.png?raw=true) | ![](https://github.com/ajulyav/FederatedLearning_Demo/blob/main/imgs/sampl1_3.png?raw=true)  |
| Sample 2 | ![](https://github.com/ajulyav/FederatedLearning_Demo/blob/main/imgs/sampl2_1.png?raw=true)  |  ![](https://github.com/ajulyav/FederatedLearning_Demo/blob/main/imgs/sampl2_2.png?raw=true) |  ![](https://github.com/ajulyav/FederatedLearning_Demo/blob/main/imgs/sampl2_3.png?raw=true) |
| Sample 3 |  ![](https://github.com/ajulyav/FederatedLearning_Demo/blob/main/imgs/sampl3_1.png?raw=true) | ![](https://github.com/ajulyav/FederatedLearning_Demo/blob/main/imgs/sampl3_2.png?raw=true)  | ![](https://github.com/ajulyav/FederatedLearning_Demo/blob/main/imgs/sampl3_3.png?raw=true)  |

##### Preparation and preprocessing pipeline: 
The general pipeline is presented in Figure 1. As it can be seen, it consists of two parts: data preparation and pre-processing. In the first part, each client downloads the dataset from the source, cleans (in order to delete problematic cases i.e. missing slice, image/label mismatch), and performs thresholding to combine the two label values to get the mask of the prostate. As an optional step, it requires channel selection to take T2w for the MSD dataset.

![](https://github.com/ajulyav/FederatedLearning_Demo/blob/main/imgs/pipeline.png?raw=true)
The second part represents the steps coming with FL training and they are a simple normalization pipeline: 

    1) Voxel normalization to 0.3x0.3x1.0.
    2) Changing the input image’s orientation into the “RAS” axcodes.
    3) Z-score intensity normalization.

To sum up, these steps were proposed in [6] and we follow them as an easy and straightforward solution.  

### 1.2 Networks 


### References

    1. Ferlay J, Soerjomataram I, Dikshit R, Eser S, Mathers C, Rebelo M, Parkin DM, Forman D, Bray F. 2015. Cancer incidence and mortality worldwide: Sources, methods and major patterns in GLOBOCAN 2012. Int J Cancer 136: E359–E386.
    2. Global Burden of Disease Cancer Collaboration. 2016. Global, regional, and national cancer incidence, mortality, years of life lost, years lived with disability, and disability-adjusted life-years for 32 cancer groups, 1990 to 2015: A systematic analysis for the global burden of disease study. JAMA Oncol 3: 524–548.
    3. G. Lemaitre, R. Marti, J. Freixenet, J. C. Vilanova, P. M. Walker, and F. Meriaudeau, "Computer-Aided Detection and Diagnosis for prostate cancer based on mono and multi-parametric MRI: A Review", Computer in Biology and Medicine, vol. 60, pp 8 - 31, 2015
    4. Simpson, A.L., Antonelli, M., Bakas, S., Bilello, M., Farahani, K., Ginneken, B.V., Kopp-Schneider, A., Landman, B.A., Litjens, G.J., Menze, B.H., Ronneberger, O., Summers, R.M., Bilic, P., Christ, P.F., Do, R.K., Gollub, M.J., Golia-Pernicka, J., Heckers, S., Jarnagin, W.R., McHugo, M., Napel, S., Vorontsov, E., Maier-Hein, L., & Cardoso, M.J. (2019). A large annotated medical image dataset for the development and evaluation of segmentation algorithms.
    5. Bloch N, Madabhushi A, Huisman H, Freymann J, Kirby J, Grauer M, Enquobahrie A, Jaffe C, Clarke L, Farahani K. (2015). NCI-ISBI 2013 Challenge: Automated Segmentation of Prostate Structures. 
    6. NVFlare GitHub. Available at: https://github.com/NVIDIA/NVFlare/tree/dev/examples (Accessed: 20.12.2022).

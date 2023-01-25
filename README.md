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
**ResUNet** presents the modification of the well-known segmentation network Unet (encoder-decoder type architecture, as illustrated in Figure 2 (a) replacing the convolution layers in each block with residual units (Figure 2 (b)). It has been demonstrated that the residual unit enhances network depth diversity, resulting in enhanced performance over the conventional convolution layer [7].

<p align="center">
  <img width="360" height="250" src="https://github.com/ajulyav/FederatedLearning_Demo/blob/main/imgs/resunet.png">
</p>

<p align="center">
Figure 2 – Modifications of the classical Unet model [7]
</p>

In global, the whole architecture is presented in Figure 3.
<p align="center">
  <img width="360" height="400" src="https://github.com/ajulyav/FederatedLearning_Demo/blob/main/imgs/resunet1.jpg">
</p>

<p align="center">
Figure 3 – ResUNet: schematic visualization of the network architecture 
</p>

**Attention Unet** is the conventional U-Net architecture incorporating the additive attention gate (AG) to draw attention to important features passed over skip connections [8].
<p align="center">
  <img width="760" height="300" src="https://github.com/ajulyav/FederatedLearning_Demo/blob/main/imgs/attention.png">
</p>

<p align="center">
Figure 4 – Attention Unet: schematic visualization of the network architecture [8]
</p>

**SegResnet** with an encoder to extract image characteristics and a decoder to reconstruct the segmentation mask, this model uses a CNN architecture based on an encoder-decoder pipeline. The created model, with the exception of the variational autoencoder (VAE) is presented in Myronenko, Andriy. (2019) [9]: 
<p align="center">
  <img width="500" height="250" src="https://github.com/ajulyav/FederatedLearning_Demo/blob/main/imgs/segres.png">
</p>

<p align="center">
Figure 5 – SegResNet: schematic visualization of the network architecture 
</p>

#### Training Procedure

In general, the same augmentation and pre-processing steps as well as hyperparameters proposed are applied as in [4]. The only changes are related to the total number of global rounds (reduced to 100). This decision allows us to reduce the total time of experiments while obtaining optimal segmentation accuracy.  

The following augmentation techniques were applied on the fly during training: random cropping of fixed sized 4 regions (224 x  224 x 32) with 1 to 1 ratio choosing foreground/background pixels, random flips along each axis, random intensity scaling and shifting.

Due to the patch-based nature of our training, all inference is done patch-based as well. In particular, a sliding window is chosen with the size of 224 x  224 x 32 and overlapping of 0.25. 

The accuracy segmentation metric is average Dice score, the SGD optimizer with learning rate of 1e-2  and momentum=0.9 is used for all experiments. The number of global rounds is set to 100 while the local epochs equal 10.   

### 1.3 Federated Learning: Algorithms
The following aggregation algorithms are tested in the framework of Demonstrator 1: 

    • Algorithm 1: FedAvg – the Federated Averaging (FedAvg) algorithm, which consists of alternating between a few local stochastic gradient updates at client nodes, followed by a model averaging update at the server, is perhaps the most commonly used method in Federated Learning [10]. 
<p align="center">
  <img width="380" height="250" src="https://github.com/ajulyav/FederatedLearning_Demo/blob/main/imgs/algo1.png">
</p>

<p align="center">
Figure 6 – FedAvg algorithm [11]
</p>

Algorithm 2: FedProx is a generalization of FedAvg with some modifications to address heterogeneity of data and systems. Differently from FedAvg, here the clients optimize a regularized loss with a proximal term. (Note: FedAvg is a particular case of FedProx with μ=0.) [11]. 

<p align="center">
  <img width="380" height="250" src="https://github.com/ajulyav/FederatedLearning_Demo/blob/main/imgs/algo2.png">
</p>

<p align="center">
Figure 7 – FedProx algorithm [11]
</p>

Algorithm 3: Ditto fundamentally differs from FedProx in that the goal is to learn personalized models vk, while FedProx produces a single global model w. For instance, when the regularization hyper-parameter is zero, Ditto reduces to learning separate local models, whereas FedProx would reduce to FedAvg [12].

<p align="center">
  <img width="380" height="350" src="https://github.com/ajulyav/FederatedLearning_Demo/blob/main/imgs/algo3.png">
</p>

<p align="center">
Figure 8 – Ditto algorithm [12]
</p>

### 2. Infrastructure: Confidential part

### 3. Framework
NVIDIA FLARE [13] is selected to be the framework as it is a domain-agnostic, open-source, and extensible SDK for Federated Learning. It enables developers to create a secure, privacy-preserving solution for a distributed multi-party cooperation and helps academics and data scientists to adapt current ML/DL workflow to a federated paradigm.

With its componentized design, NVIDIA FLARE's federated learning workloads may be easily moved from research and simulation to actual production deployment. Some essential elements include:

- FL Simulator for rapid development and prototyping (new in v2.2)
- FLARE Dashboard for simplified project management and deployment (new in v2.2)
- Reference FL algorithms (e.g., FedAvg, FedProx) and workflows (e.g., Scatter and Gather, Cyclic)
- Privacy preservation with differential privacy, homomorphic encryption, and more
- Management tools for secure provisioning and deployment, orchestration, and management
-Specification-based API for extensibility
    
<p align="center">
  <img width="680" height="350" src="https://github.com/ajulyav/FederatedLearning_Demo/blob/main/imgs/flare.png">
</p>

<p align="center">
Figure 11 – NVIDIA FLARE key features
</p>

As at the date of starting of the experiments, there were two available versions 2.0 and 2.1. To continue experiments, NVIDIA FLARE 2.0 was chosen as the simple scenario where there is one experiment per time and a single server-aggregator. This setup is fully supported by 2.0 and removes the need for using NVIDIA FLARE 2.1 which mainly provides the updates of: High Availability (i.e. support of multiple FL servers with automatic cut-over) and Multi-Job Execution (i.e. run of multiple experiments in parallel).

### 4. Experiments 
As mentioned before, since the purpose of the demonstrator is to test the feasibility of the Partners to run FL workflows remotely the FL workflows (experiments), the main focus is on baseline experiments:

• Test FL average techniques during training.

• Measure segmentation accuracy.

• Evaluate training time.

• Compare FL pipelines against local training.


Thus, Table 3 presents the setup plan of all experiments being conducted for Demonstrator 1. 

Table 3 – Experiment plan of Demonstrator 1
<p align="center">
  <img width="580" height="450" src="https://github.com/ajulyav/FederatedLearning_Demo/blob/main/imgs/table.png">
</p>

These experiments allow us pursue the following goals:

<p align="center">
  <img width="580" height="90" src="https://github.com/ajulyav/FederatedLearning_Demo/blob/main/imgs/table1.png">
</p>
• Run 1, Run 2 and Run 3 (Goal 1) are implemented for the comparisons of hardware variations in the federation. As the well-known critical aspect for efficient training DL algorithms is the availability of high-performing resources, the goal is to see the effect of CPU/GPU client setups on the total execution time. 

• Though Run 3, Run 4 and Run 5 (Goal 2) are for analyzing Federated Learning and Deep Learning results, in particular, analysis of different aggregation schemes and CNN models and the overall accuracy under multi-centric data with different variations. 

• Finally, to evaluate the value of Federated Learning vs classical local training (Goal 3), the FL global models and training performed only in one center with cross-validation are compared. 


### References

    1. Ferlay J, Soerjomataram I, Dikshit R, Eser S, Mathers C, Rebelo M, Parkin DM, Forman D, Bray F. 2015. Cancer incidence and mortality worldwide: Sources, methods and major patterns in GLOBOCAN 2012. Int J Cancer 136: E359–E386.
    2. Global Burden of Disease Cancer Collaboration. 2016. Global, regional, and national cancer incidence, mortality, years of life lost, years lived with disability, and disability-adjusted life-years for 32 cancer groups, 1990 to 2015: A systematic analysis for the global burden of disease study. JAMA Oncol 3: 524–548.
    3. G. Lemaitre, R. Marti, J. Freixenet, J. C. Vilanova, P. M. Walker, and F. Meriaudeau, "Computer-Aided Detection and Diagnosis for prostate cancer based on mono and multi-parametric MRI: A Review", Computer in Biology and Medicine, vol. 60, pp 8 - 31, 2015
    4. Simpson, A.L., Antonelli, M., Bakas, S., Bilello, M., Farahani, K., Ginneken, B.V., Kopp-Schneider, A., Landman, B.A., Litjens, G.J., Menze, B.H., Ronneberger, O., Summers, R.M., Bilic, P., Christ, P.F., Do, R.K., Gollub, M.J., Golia-Pernicka, J., Heckers, S., Jarnagin, W.R., McHugo, M., Napel, S., Vorontsov, E., Maier-Hein, L., & Cardoso, M.J. (2019). A large annotated medical image dataset for the development and evaluation of segmentation algorithms.
    5. Bloch N, Madabhushi A, Huisman H, Freymann J, Kirby J, Grauer M, Enquobahrie A, Jaffe C, Clarke L, Farahani K. (2015). NCI-ISBI 2013 Challenge: Automated Segmentation of Prostate Structures. 
    6. NVFlare GitHub. Available at: https://github.com/NVIDIA/NVFlare/tree/dev/examples (Accessed: 20.12.2022).
    7. Xue W, Li J, Hu Z, Kerfoot E, Clough J, Oksuz I, Xu H, Grau V, Guo F, Ng M, Li X, Li Q, Liu L, Ma J, Grinias E, Tziritas G, Yan W, Atehortua A, Garreau M, Jang Y, Debus A, Ferrante E, Yang G, Hua T, Li S. Left Ventricle Quantification Challenge: A Comprehensive Comparison and Evaluation of Segmentation and Regression for Mid-Ventricular Short-Axis Cardiac MR Data. IEEE J Biomed Health Inform. 2021 Sep;25(9):3541-3553. doi: 10.1109/JBHI.2021.3064353. Epub 2021 Sep 3. PMID: 33684050; PMCID: PMC7611810.
    8. Oktay, Ozan & Schlemper, Jo & Folgoc, Loic & Lee, Matthew & Heinrich, Mattias & Misawa, Kazunari & Mori, Kensaku & McDonagh, Steven & Hammerla, Nils & Kainz, Bernhard & Glocker, Ben & Rueckert, Daniel. (2018). Attention U-Net: Learning Where to Look for the Pancreas. 
    9. Myronenko, Andriy. (2019). 3D MRI Brain Tumor Segmentation Using Autoencoder Regularization: 4th International Workshop, BrainLes 2018, Held in Conjunction with MICCAI 2018, Granada, Spain, September 16, 2018, Revised Selected Papers, Part II. 10.1007/978-3-030-11726-9_28. 
    10. McMahan, H. B., Eider Moore, Daniel Ramage, Seth Hampson and Blaise Agüera y Arcas. “Communication-Efficient Learning of Deep Networks from Decentralized Data.” International Conference on Artificial Intelligence and Statistics (2017).
    11. Sahu, Anit Kumar, Tian Li, Maziar Sanjabi, Manzil Zaheer, Ameet S. Talwalkar and Virginia Smith. “Federated Optimization in Heterogeneous Networks.” arXiv: Learning (2020): n. pag.
    12. Li, Tian, Shengyuan Hu, Ahmad Beirami and Virginia Smith. “Ditto: Fair and Robust Federated Learning Through Personalization.” International Conference on Machine Learning (2021).
    13. NVFlare Developer. Available at: https://developer.nvidia.com/flare (Accessed: 20.12.2022).

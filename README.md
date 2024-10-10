# _NPDS_ for nodule progression detection
This repository contains the code for [**A Novel Statistic Guided by Clinical Experience for Assessing the Progression of Lung Nodules**](url).

## Contents
- [Use Terms](#use-terms)
- [Introduction](#introduction)
- [Methods](#methods)
- [Requirements](#requirements)
- [Installation](#installation)

## Use Terms

### Intellectual Property and Rights Notice
All content within this repository, including but not limited to source code, models, algorithms, data, and documentation, are subject to applicable intellectual property laws. The rights to this project are reserved by the project's author(s) or the rightful patent holder(s).

### Limitations on Commercial Use
This repository's contents, protected by patent, are solely for personal learning and research purposes, and are not for commercial use. Any organizations or individuals must not use any part of this project for commercial purposes without explicit written permission from the author(s) or rightful patent holder(s). Violations of this restriction will result in legal action.

### Terms for Personal Learning and Academic Research
Individual users are permitted to use this repository for learning and research purposes, provided that they abide by applicable laws. Should you utilize this project in your research, please cite our work as follows:

>Zhou, J., Yu, H., Wang, H., and Ji, Y. A Novel Statistic Guided by Clinical Experience for Assessing the Progression of Lung Nodules. doi

## Introduction

Pulmonary subsolid nodules (SSNs) usually have a high malignant potential but are slow-growing. Diagnosing SSNs as benign or malignant only based on a baseline CT scan is difficult, so regular follow-up scans are crucial in clinical practice. Traditionally, with the follow-up CT scans, physicians can evaluate the progression or changes in SSNs by measuring their size, density, or volume. However, these assessments are usually subjective and can be influenced by physician experience and external factors.

In this study, we innovatively consider the nodule progression problem from a hypothesis testing perspective and propose a novel _NPDS_ statistic to determine whether a SSN has progressed or not. This method relies only on 2D CT images, avoiding the complex process of traditional volumetric methods that require precise 3D nodule delineation. In the meanwhile, the proposed statistic ensures that the assessment of nodule progression accounts for both statistical significance and physicians' clinical experience, making it more valuable for clinical application.

## Methods

### Workflow for _NPDS_ construction
Fig 1 shows the construction workflow for _NPDS_, where each step is guided by clinical experience.

![NPDS_construction_flow.png](https://github.com/hangyustat/NPDS/blob/main/Images/NPDS_construction_flow.png)
### Hypothesis testing
The null hypothesis (i.e., $H_{0}$) is that there is no change in the nodule areas between the two CT scans, while the alternative hypothesis (i.e., $H_{1}$) is that there is a change in the nodule areas between the two CT scans.

To obtain the distribution of the _NPDS_ statistic under the null hypothesis, we generated a large-scale clinically invariant nodule sample (ClinvNod sample) by two-step observational study and random perturbation method. Then, the hypothesis testing could be conducted under the _NPDS_ sampling distribution of ClinvNod sample.

Fig 2 shows the design of the two-step observational study.

![two_step_ob_study.png](https://github.com/hangyustat/NPDS/blob/main/Images/two_step_ob_study.png)

Fig 3 shows the sampling distribution of _NPDS_ for one-tailed (left) and two-tailed (right) test.

![NPDS_sampling_distribution.png](https://github.com/hangyustat/NPDS/blob/main/Images/NPDS_sampling_distribution.png)

### How to construct your own _NPDS_?
Prepare your own data according to the instructions in NPDS_calculate.ipynb, and rerun it.
## Requirements

The code is written in Python and requires the following packages: 

* Python 3.8 
* Numpy 1.24.4 
* Pandas 2.0.3 
* Matplotlib 3.6.3 
* Scipy 1.8.0
* Skimage 0.21.0
* Itk 5.4.0
* Plotnine 0.12.4
* Ipywidgets 7.6.5
* Pillow 10.0.0
* Ipython 7.30.1
## Installation
* Install Python 3.8
* pip install -r requirements.txt

# _NPDS_ statistic for nodule progression detection
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

>doi

## Introduction

Pulmonary subsolid nodules (SSNs) usually have a high malignant potential but are slow-growing. Diagnosing SSNs as benign or malignant only based on a baseline CT scan is difficult, so regular follow-up scans are crucial in clinical practice. Traditionally, with the follow-up CT scans, physicians can evaluate the progression or changes in SSNs by measuring their size, density, or volume. However, these assessments are usually subjective and can be influenced by physician experience and external factors.

In this study, we innovatively consider the nodule progression problem from a hypothesis testing perspective and propose a novel _NPDS_ statistic to determine whether a SSN has progressed or not. This method relies only on 2D CT images, avoiding the complex process of traditional volumetric methods that require precise 3D nodule delineation. In the meanwhile, the proposed statistic ensures that the assessment of nodule progression accounts for both statistical significance and physicians' clinical experience, making it more valuable for clinical application.

## Methods

### Workflow for _NPDS_ construction
Fig 1 shows the construction workflow for _NPDS_, where each step is guided by clinical experience.
### Hypothesis testing
The null hypothesis (i.e., $H_{0}$) is that there is no change in the nodule areas between the two CT scans, while the alternative hypothesis (i.e., $H_{1}$) is that there is a change in the nodule areas between the two CT scans.

To obtain the distribution of the _NPDS_ statistic under the null hypothesis, we generated a large-scale clinically invariant nodule sample (ClinvNod sample) using random perturbation method and conducted hypothesis testing under its NPDS empirical distribution.

Fig 2 shows the sampling distribution of _NPDS_ for one-tailed (left) and two-tailed (right) test.

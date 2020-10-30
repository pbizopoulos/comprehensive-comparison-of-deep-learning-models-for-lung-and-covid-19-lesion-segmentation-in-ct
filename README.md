[![citation](http://img.shields.io/badge/Citation-0091FF.svg)](https://scholar.google.com/scholar?q=Comprehensive%20Comparison%20of%20Deep%20Learning%20Models%20for%20Lung%20and%20COVID-19%20Lesion%20Segmentation%20in%20CT%20scans.%20arXiv%202020)
[![arXiv](http://img.shields.io/badge/eess.IV-arXiv%3A2009.06412-B31B1B.svg)](https://arxiv.org/abs/2009.06412)

# Comprehensive Comparison of Deep Learning Models for Lung and COVID-19 Lesion Segmentation in CT
This repository contains the code that generates the results of the paper **Comprehensive Comparison of Deep Learning Models for Lung and COVID-19 Lesion Segmentation in CT**.

## Requirements
- UNIX tools (awk, cut, grep)
- docker
- make
- nvidia-container-toolkit [required only when using CUDA]

## Instructions
1. `git clone https://github.com/pbizopoulos/comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-in-ct`
2. `cd comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-in-ct`
3. `sudo systemctl start docker`
4. `make help`

## Instructions for evaluating the trained models
1. `git clone https://github.com/pbizopoulos/comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-in-ct`
2. `cd comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-in-ct/web-app`
3. `make`
4. visit http://0.0.0.0:7860/ in your browser

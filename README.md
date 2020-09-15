[![citation](http://img.shields.io/badge/Citation-0091FF.svg)](https://scholar.google.com/scholar?q=)
[![arXiv](http://img.shields.io/badge/eess-arXiv%3A2009.06412-B31B1B.svg)](https://arxiv.org/abs/2009.06412)

# Comprehensive Comparison of Deep Learning Models for Lung and COVID-19 Lesion Segmentation in CT
This repository contains the code that generates the results of the paper **Comprehensive Comparison of Deep Learning Models for Lung and COVID-19 Lesion Segmentation in CT**.

## Requirements
- UNIX tools (awk, cut, grep)
- docker
- make
- nvidia-container-toolkit [required only when using CUDA]

## Instructions for verifying the reproducibility of this paper
1. `git clone https://github.com/pbizopoulos/comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-in-ct`
2. `cd comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-in-ct`
3. `sudo systemctl start docker`
4. `make test`

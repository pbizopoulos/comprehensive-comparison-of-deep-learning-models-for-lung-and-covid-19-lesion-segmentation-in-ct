[![arXiv](http://img.shields.io/badge/eess.IV-arXiv%3A2009.06412-B31B1B.svg)](https://arxiv.org/abs/2009.06412)
[![citation](http://img.shields.io/badge/citation-0091FF.svg)](https://scholar.google.com/scholar?q=Comprehensive%20Comparison%20of%20Deep%20Learning%20Models%20for%20Lung%20and%20COVID-19%20Lesion%20Segmentation%20in%20CT%20scans.%20arXiv%202020)
[![template](http://img.shields.io/badge/template-EEE0B1.svg)](https://github.com/pbizopoulos/docker-as-a-development-environment-for-research-papers-template)
[![test-local-reproducibility](https://github.com/pbizopoulos/comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-in-ct/workflows/test-local-reproducibility/badge.svg)](https://github.com/pbizopoulos/comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-in-ct/actions?query=workflow%3Atest-local-reproducibility)

# Comprehensive Comparison of Deep Learning Models for Lung and COVID-19 Lesion Segmentation in CT
This repository contains the code that generates the paper **Comprehensive Comparison of Deep Learning Models for Lung and COVID-19 Lesion Segmentation in CT**.

## Requirements
- UNIX utilities (cmp, cp, echo, rm, touch)
- [docker](https://docs.docker.com/get-docker/)
- make
- [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit) (required only when using CUDA)

## Instructions
1. `git clone https://github.com/pbizopoulos/comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-in-ct`
2. `cd comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-in-ct`
3. `sudo systemctl start docker`
4. make options
    * `make`             # Generate pdf.
    * `make ARGS=--full` # Generate full pdf.
    * `make clean`       # Remove build and cache directories.

## Instructions for evaluating the trained models
1. `git clone https://github.com/pbizopoulos/comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-in-ct`
2. `cd comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-in-ct/web-app`
3. `sudo systemctl start docker`
4. `make`
5. visit http://0.0.0.0:7860/ in your browser

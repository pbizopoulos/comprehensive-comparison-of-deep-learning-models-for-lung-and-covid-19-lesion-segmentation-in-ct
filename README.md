[![arXiv](http://img.shields.io/badge/eess.IV-arXiv%3A2009.06412-B31B1B.svg)](https://arxiv.org/abs/2009.06412)
[![citation](http://img.shields.io/badge/citation-0091FF.svg)](https://scholar.google.com/scholar?q=Comprehensive%20Comparison%20of%20Deep%20Learning%20Models%20for%20Lung%20and%20COVID-19%20Lesion%20Segmentation%20in%20CT%20scans.%20arXiv%202020)
[![template](http://img.shields.io/badge/template-EEE0B1.svg)](https://github.com/pbizopoulos/a-makefile-for-developing-containerized-latex-technical-documents-template)
[![test-draft-version-document](https://github.com/pbizopoulos/comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-in-ct/workflows/test-draft-version-document/badge.svg)](https://github.com/pbizopoulos/comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-in-ct/actions?query=workflow%3Atest-draft-version-document)


# Comprehensive Comparison of Deep Learning Models for Lung and COVID-19 Lesion Segmentation in CT
This repository contains the code that generates **Comprehensive Comparison of Deep Learning Models for Lung and COVID-19 Lesion Segmentation in CT**.

## Requirements
- [POSIX-oriented operating system](https://en.wikipedia.org/wiki/POSIX#POSIX-oriented_operating_systems)
- [Docker](https://docs.docker.com/get-docker/)
- [Make](https://www.gnu.org/software/make/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit) (required only when using CUDA)

## Instructions
1. `git clone https://github.com/pbizopoulos/comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-in-ct`
2. `cd comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-in-ct/`
3. `sudo systemctl start docker`
4. make options
    * `make`                # Generate the draft (fast) version document.
    * `make VERSION=--full` # Generate the full (slow) version document.
    * `make clean`          # Remove the tmp/ directory.

## Instructions for evaluating trained models using [Gradio](https://github.com/gradio-app/gradio)
1. `git clone https://github.com/pbizopoulos/comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-in-ct`
2. `cd comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-in-ct/demos/`
3. `sudo systemctl start docker`
4. `make`
5. visit http://0.0.0.0:7860/ in your browser

## Instructions for evaluating trained models using [Jupyter](https://jupyter.org/)
1. visit https://github.com/pbizopoulos/comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-in-ct/blob/master/demos/main.ipynb

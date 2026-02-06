---
title: "PyNNLF: Python for Network Net Load Forecasting"
tags:
  - Python
  - forecasting
  - net load
  - evaluation
  - dataset library
  - model library
authors:
  - name: M. Syahman Samhan
    affiliation: [1, 2]
  - name: Anna Bruce
    affiliation: [1, 2]
  - name: Baran Yildiz
    affiliation: [1, 2]
affiliations:
  - name: School of PV and Renewable Energy Engineering, UNSW Sydney, Australia
    index: 1
  - name: Collaboration on Energy and Environmental Markets (CEEM), UNSW Sydney, Australia
    index: 2
date: 18 September 2025
bibliography: paper.bib
citeproc: true
---

# Statement of need
As solar photovoltaic (PV) system installations increase, network operators must forecast net
electricity load—the difference between electricity consumption and PV generation. Since 2016 [@Kaur_2016], more than 100 net load forecasting papers have been published, and most (84 papers) introduced new models and claimed superior performance [@Tziolis_2025]. Typical statements include:

| Statement                                                                                                                        | Reference         |
|----------------------------------------------------------------------------------------------------------------------------------|-------------------|
| … and it is concluded that the proposed method has higher prediction accuracy and better prediction effect …                    | [@Cao_2023]       |
| Comparative tests utilizing real-world data verify the superiority of the proposed method over other state-of-the-art algorithms | [@Hu_2024]        |
| The performance of the BDLSTM model dominates when compared with the best of the state-of-the-art methods …                     | [@Sun_2020]       |

However, limited attention has been given to reliability and reproducibility: 81% did not compare against simple benchmark models such as naive or mean models, 60% evaluated the models only on one dataset, and 94% did not make the code publicly available. This creates a clear need for a tool that evaluates net load forecasting models reliably and reproducibly. Such a tool should include a library of publicly available net load datasets and models, and enable users to add their own datasets and models to compare against established baselines.

# Summary

PyNNLF (Python for Network Net Load Forecasting) is an open-source tool designed to enable reliable and reproducible evaluation of net load forecasting models. It includes:

A library of commonly used net load datasets (e.g., Ausgrid Solar Home Data [@Ausgrid_2014]), and a collection of 18 forecasting models, ranging from simple benchmarks (e.g., `naïve model`) to statistical models (e.g., `linear regression`) and machine learning models (e.g., `artificial neural networks`).


The PyNNLF software is available as an open-source repository on GitHub [here](https://github.com/mssamhan31/PyNNLF) [@PyNNLF_Repo]. Comprehensive documentation is provided [here](https://mssamhan31.github.io/PyNNLF/) [@PyNNLF_Docs].

Users can specify the forecasting problem (dataset and forecast horizon) and model configuration (model name and hyperparameters). PyNNLF then outputs evaluation results including performance metrics, metadata, visualizations, and supplemental outputs.

Researchers and network operators can use PyNNLF to benchmark their models against others using standardized datasets. They can also contribute new models or datasets to the PyNNLF library, enabling broader comparison and collaboration While libraries like `statsmodels`, `PyTorch`, or `Darts` allow some model comparison with shared data and metrics, they are not designed specifically for net load forecasting. They lack curated datasets and models for this purpose and do not offer a clear way to record experiments. PyNNLF addresses these gaps by providing a focused framework with integrated datasets and structured experiment tracking.

# Research field
Net load forecasting sits within energy forecasting and energy systems research. While general-purpose forecasting tools exist, they are not tailored to net load forecasting and do not provide curated datasets, baseline models, and standardized experiment outputs in a single workflow.

# Software design
PyNNLF is a Python package that is installed using `pip install pynnlf`. It provides a reproducible workflow based on YAML specifications, standardized experiment outputs, and modular model and dataset discovery.

# Research impact statement
By standardizing datasets, baselines, and reporting, PyNNLF makes net load forecasting research more reliable and reproducible and enables fair comparisons across studies.

# AI usage
We used AI for simple editing assistance (spelling, grammar, etc.) and coding assistance, especially for syntax writing with GitHub Copilot with the GPT-5.2-Codex model. The overall architecture of the software and user requirements were defined by the authors. 

In parallel with developing PyNNLF, we are also preparing other research papers: a literature review of net load forecasting studies, and comparative analyses of various models on multiple net load datasets, forecast horizons, spatial aggregations for the load, and minimum demand forecasting using PyNNLF.

# Acknowledgements
This research is part of Samhan’s PhD study, which is sponsored by University International Postgraduate Award (UIPA) UNSW scholarship [@UNSW_2025] and industry collaboration partnership with Ausgrid [@Ausgrid_2025], a Distribution Network Service Provider in Australia, and RACE For 2030 Scholarship.

# References
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

# Summary
As solar photovoltaic (PV) system installations increase, network operators must forecast not only electricity load but net electricity load—the difference between electricity consumption and PV generation. The term net load forecasting was first introduced in an academic paper in 2016 [@Kaur_2016]. Since then, over 60 journal articles and conference papers have been published on the topic by 2025 [@Tziolis_2025]. Most focus on proposing new, complex models and claiming superior performance. Typical statements include:

| Statement                                                                                                                        | Reference         |
|----------------------------------------------------------------------------------------------------------------------------------|-------------------|
| … and it is concluded that the proposed method has higher prediction accuracy and better prediction effect …                    | [@Cao_2023]       |
| Comparative tests utilizing real-world data verify the superiority of the proposed method over other state-of-the-art algorithms | [@Hu_2024]        |
| The performance of the BDLSTM model dominates when compared with the best of the state-of-the-art methods …                     | [@Sun_2020]       |



However, around 75% of these studies did not use simple benchmarks such as the `naïve model`, which forecasts the next value as equal to the last observation. Additionally, 63% did not use public datasets, and 99% did not share their code. This indicates a strong focus on improving forecasting accuracy, but limited attention to standardizing evaluation process, model reliability and reproducibility.
PyNNLF (Python for Network Net Load Forecasting) is an open-source tool designed to address these gaps by enabling reliable and reproducible evaluation of net load forecasting models. It includes:

A library of commonly used net load datasets (e.g., Ausgrid Solar Home Data [@Ausgrid_2014]), and a collection of 18 forecasting models, ranging from simple benchmarks (e.g., `naïve model`) to statistical models (e.g., `linear regression`) and machine learning models (e.g., `artificial neural networks`).


The PyNNLF software is available as an open-source repository on GitHub [here](https://github.com/mssamhan31/PyNNLF) [@PyNNLF_Repo]. Comprehensive documentation is provided [here](https://mssamhan31.github.io/PyNNLF/) [@PyNNLF_Docs].

Users can specify the forecasting problem (dataset and forecast horizon) and model configuration (model name and hyperparameters). PyNNLF then outputs evaluation results including performance metrics, metadata, visualizations, and supplemental outputs.
Researchers and network operators can use PyNNLF to benchmark their models against others using standardized datasets. They can also contribute new models or datasets to the PyNNLF library, enabling broader comparison and collaboration.
While general time series forecasting libraries like `statsmodels`, `PyTorch`, or `Darts` exist, none specifically focus on net load forecasting with curated datasets and models.
In parallel with developing PyNNLF, we are also preparing other research papers: a literature review of net load forecasting studies, and comparative analyses of various models on multiple datasets, forecast horizons, spatial aggregations, and minimum demand forecasting using PyNNLF.

# Acknowledgements
This research is part of Samhan’s PhD study, which is sponsored by University International Postgraduate Award (UIPA) UNSW scholarship [@UNSW_2025] and industry collaboration partnership with Ausgrid [@Ausgrid_2025], a Network Service Provider in Australia.

# References
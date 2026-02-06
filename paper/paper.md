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
PyNNLF (Python for Network Net Load Forecasting) is an open-source tool for reliable and reproducible evaluation of net load forecasting models. It provides curated net load datasets and a library of 18 forecasting models, from simple benchmarks (e.g. naive model) to statistical (e.g. linear regression) and machine learning (e.g. artificial neural network) approaches. Users specify a forecast problem and model configuration, and PyNNLF produces standardized evaluation outputs, plots, and metadata to support fair comparisons across models.

# Statement of need
As solar photovoltaic (PV) system installations increase, network operators must forecast net electricity load, the difference between electricity consumption and PV generation. Since 2016 [@Kaur_2016], more than 100 net load forecasting papers have been published, and most (84 papers) introduced new models and claimed superior performance [@Tziolis_2025]. Typical statements include:

| Statement                                                                                                                        | Reference         |
|----------------------------------------------------------------------------------------------------------------------------------|-------------------|
| ... and it is concluded that the proposed method has higher prediction accuracy and better prediction effect ...                | [@Cao_2023]       |
| Comparative tests utilizing real-world data verify the superiority of the proposed method over other state-of-the-art algorithms | [@Hu_2024]        |
| The performance of the BDLSTM model dominates when compared with the best of the state-of-the-art methods ...                   | [@Sun_2020]       |

However, limited attention has been given to reliability and reproducibility: 81% did not compare against simple benchmark models such as naive or mean models, 60% evaluated models only on one dataset, and 94% did not make code publicly available. This creates a clear need for a tool that evaluates net load forecasting models reliably and reproducibly for researchers and industry practitioners working in net load forecasting. Such a tool should include a library of publicly available net load datasets and models, and enable users to add their own datasets and models to compare against established baselines.

# State of the field
Net load forecasting sits within energy forecasting and energy systems research. Existing alternatives such as statsmodels, PyTorch, or Darts are valuable for general forecasting, but they do not provide a centralized place to host a net load dataset library and commonly used net load models. They also do not offer standardized experiment outputs tailored to net load forecasting. PyNNLF fills this gap by combining curated datasets, benchmark models, and structured experiment tracking in one workflow.

# Software design
PyNNLF prioritizes functionality and usability. It provides a library of commonly used datasets and models, supports comparison across models, and allows users to specify hyperparameters, add new models, and add new datasets. The API is intentionally simple and exposed through a Python package that can be installed with `pip install pynnlf`. The backend handles cross validation, evaluation, and result summarization so users can focus on model comparison and model creation rather than experimental plumbing.

# Research impact statement
By standardizing datasets, baselines, and reporting, PyNNLF makes net load forecasting research more reliable and reproducible and enables fair comparisons across studies.

# AI usage disclosure
We used AI for simple editing assistance (spelling and grammar) and for coding assistance, especially with syntax. The overall architecture of the software and the user requirements were defined by the authors. We used GitHub Copilot with the GPT-5.2-Codex model, and all AI-proposed code and actions were manually reviewed and verified to ensure correctness and preserve software functionality.

# Acknowledgements
This research is part of Samhan’s PhD study, which is sponsored by University International Postgraduate Award (UIPA) UNSW scholarship [@UNSW_2025] and industry collaboration partnership with Ausgrid [@Ausgrid_2025], a Distribution Network Service Provider in Australia, and RACE For 2030 Scholarship..

# References
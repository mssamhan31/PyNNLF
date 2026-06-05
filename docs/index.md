---
hide:
  - navigation
  - toc
---

<section class="pynnlf-landing">
  <header class="pynnlf-hero">
    <p class="pynnlf-eyebrow">PyNNLF</p>
    <h1>Reliable net load forecasting evaluation, not just another new model.</h1>
    <p class="pynnlf-hero__lead">
      PyNNLF (Python for Network Net Load Forecasting) is an open-source Python tool for comparing net load forecasting models with public datasets, simple benchmarks, cross-validation, and reproducible experiment outputs.
    </p>
    <p class="pynnlf-definition">
      Net load is the underlying electricity load minus renewable energy generation. Net load forecasting means predicting that remaining demand over a future forecast horizon.
    </p>
    <div class="pynnlf-actions">
      <a class="pynnlf-button pynnlf-button--primary pynnlf-button--install" href="getting_started/">Install PyNNLF</a>
      <a class="pynnlf-button" href="examples/">Read the docs</a>
      <a class="pynnlf-button" href="https://github.com/mssamhan31/PyNNLF">GitHub</a>
    </div>
  </header>

  <section class="pynnlf-section pynnlf-problem">
    <div class="pynnlf-section__intro">
      <p class="pynnlf-eyebrow">The research issue</p>
      <h2>Many papers claim superior forecasting accuracy. Fewer make the comparison easy to trust.</h2>
      <p>
        Since 2016, more than 102 academic papers have been published on net load forecasting. At least 84 introduced a novel model. However, many papers did not use simple benchmark models, relied on private datasets, or did not publicly share implementation code.
      </p>
    </div>

    <div class="pynnlf-quote-stack">
      <p class="pynnlf-quote-lead">Typical excerpts found in net load forecasting papers include:</p>
      <figure>
        <blockquote>&ldquo;&hellip; and it is concluded that the proposed method has higher prediction accuracy and better prediction effect &hellip;&rdquo;</blockquote>
        <figcaption><a href="https://doi.org/10.1088/1742-6596/2418/1/012069">[1] Cao et al., 2023</a></figcaption>
      </figure>
      <figure>
        <blockquote>&ldquo;Comparative tests utilizing real-world data verify the superiority of the proposed method over other state-of-the-art net load forecasting algorithms.&rdquo;</blockquote>
        <figcaption><a href="https://doi.org/10.1016/j.renene.2024.120253">[2] Hu et al., 2024</a></figcaption>
      </figure>
      <figure>
        <blockquote>&ldquo;More-over, the performance of the BDLSTM model also dominates when compared with the best of the state-of-the-art methods, &hellip;&rdquo;</blockquote>
        <figcaption><a href="https://doi.org/10.1109/TPWRS.2019.2924294">[3] Sun et al., 2020</a></figcaption>
      </figure>
    </div>
  </section>

  <section class="pynnlf-section">
    <div class="pynnlf-stat-grid" aria-label="Literature review summary facts">
      <div>
        <strong>102+</strong>
        <span>net load forecasting papers since 2016</span>
      </div>
      <div>
        <strong>84</strong>
        <span>introduced a novel model</span>
      </div>
      <div>
        <strong>75%</strong>
        <span>did not compare with naive or seasonal naive benchmarks</span>
      </div>
      <div>
        <strong>58%</strong>
        <span>did not use a publicly available dataset</span>
      </div>
      <div>
        <strong>94%+</strong>
        <span>did not make their code publicly available</span>
      </div>
    </div>
  </section>

  <section class="pynnlf-section pynnlf-solution">
    <div class="pynnlf-section__intro">
      <p class="pynnlf-eyebrow">The tool</p>
      <h2>PyNNLF turns model comparison into a repeatable workflow.</h2>
      <p>
        Users define the forecast problem and model specification in a YAML file. PyNNLF prepares the data, creates lag and calendar features, runs cross-validation, and stores the result using a consistent output structure.
      </p>
    </div>
    <div class="pynnlf-workflow">
      <span>Dataset</span>
      <span>Forecast horizon</span>
      <span>Model and hyperparameters</span>
      <span>Cross-validated outputs</span>
    </div>
  </section>

  <section class="pynnlf-section pynnlf-capability">
    <p class="pynnlf-eyebrow">What it outputs</p>
    <h2>Accuracy, stability, runtime, plots, and trained models in one experiment folder.</h2>
    <div class="pynnlf-capability-grid">
      <article>
        <h3>Accuracy</h3>
        <p>Train and test errors, including RMSE and nRMSE.</p>
      </article>
      <article>
        <h3>Stability</h3>
        <p>Cross-validation standard deviation to show whether performance is consistent.</p>
      </article>
      <article>
        <h3>Runtime</h3>
        <p>Training time so accuracy can be weighed against computational cost.</p>
      </article>
      <article>
        <h3>Reproducibility</h3>
        <p>Fold-level forecasts, residuals, trained models, and recap files.</p>
      </article>
    </div>
  </section>

  <section class="pynnlf-section pynnlf-install">
    <div>
      <p class="pynnlf-eyebrow">Use it</p>
      <h2>Install PyNNLF with pip and run an experiment from a YAML spec.</h2>
    </div>
    <div class="pynnlf-code-pair">
      <pre><code>python -m pip install pynnlf</code></pre>
      <pre><code>import pynnlf

pynnlf.init("example_project")
pynnlf.run_experiment("example_project/specs/experiment.yaml")</code></pre>
    </div>
  </section>

  <section class="pynnlf-references" aria-label="References">
    <p>[1] H. Cao, L. Yang, H. Li, K. Wang, Net Power Prediction for High Permeability Distributed Photovoltaic Integration System, J. Phys. Conf. Ser., 2023. <a href="https://doi.org/10.1088/1742-6596/2418/1/012069">https://doi.org/10.1088/1742-6596/2418/1/012069</a>.</p>
    <p>[2] J. Hu, W. Hu, D. Cao, X. Sun, J. Chen, Y. Huang, Z. Chen, F. Blaabjerg, Probabilistic net load forecasting based on transformer network and Gaussian process-enabled residual modeling learning method, Renew. Energy 225 (2024). <a href="https://doi.org/10.1016/j.renene.2024.120253">https://doi.org/10.1016/j.renene.2024.120253</a>.</p>
    <p>[3] M. Sun, T. Zhang, Y. Wang, G. Strbac, C. Kang, Using Bayesian Deep Learning to Capture Uncertainty for Residential Net Load Forecasting, IEEE Transactions on Power Systems 35 (2020) 188-201. <a href="https://doi.org/10.1109/TPWRS.2019.2924294">https://doi.org/10.1109/TPWRS.2019.2924294</a>.</p>
  </section>

  <section class="pynnlf-disclosure" aria-label="Disclosure">
    <p>Disclosure: PyNNLF is an open-source tool developed as part of Samhan's PhD study, which is funded by UNSW Sydney, Ausgrid, RACE for 2030, and the NSW Decarbonisation Innovation Hub.</p>
  </section>
</section>

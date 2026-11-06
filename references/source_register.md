# Academic Source Register

Prepared 30 October 2026. Publication records and initial relevance were checked against publisher pages, official proceedings or author-deposited publications. Reading depth is recorded below so a verified citation is not confused with a completed critical reading. Use a source only for claims supported by the inspected material.

This shortlist contains ten academic journal or conference papers, including six published in 2022 to 2024. Those six fall within the five-year window preceding the stated November 2026 submission. They satisfy the numerical shortlist target; the finished report must actually use relevant sources, include accurate in-text citations, and critically engage with the material. The dataset and software documentation listed separately do not inflate that count.

## Sources already used in the partial draft

### R01 Bank marketing problem and source study

Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. *Decision Support Systems, 62*, 22-31. https://doi.org/10.1016/j.dss.2014.03.001

**Record checked:** [Publisher record](https://www.sciencedirect.com/science/article/pii/S016792361400061X); [institutional author manuscript](https://repositorio.iscte-iul.pt/bitstream/10071/9499/5/dss_v3.pdf).

**Reading status:** Publisher abstract and metadata inspected, alongside the selected dataset's supplied documentation. The full study's methodology has not been independently reproduced.

**Use:** Business setting and historical response-prediction problem in Section 1. Read the study's temporal evaluation discussion before extending the comparison with our protocol.

**Limit:** The public additional file differs from the study's full private data. Do not transfer its reported performance or feature set to this project. **Recent-paper quota: no.**

### R02 General learning approaches

Sarker, I. H. (2021). Machine learning: Algorithms, real-world applications and research directions. *SN Computer Science, 2*, Article 160. https://doi.org/10.1007/s42979-021-00592-x

**Record and text:** [Publisher](https://link.springer.com/article/10.1007/s42979-021-00592-x); [open article](https://pmc.ncbi.nlm.nih.gov/articles/PMC7983091/).

**Reading status:** Article overview and learning taxonomy inspected.

**Use:** Basic supervised and unsupervised distinction in Section 2; algorithm overview for later focused reading.

**Limit:** Broad review; it does not establish which classifier works best for this dataset. Published March 2021, so it is outside the five-year window for November 2026. **Recent-paper quota: no.**

### R03 Leakage and evaluation design

Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. *Patterns, 4*(9), Article 100804. https://doi.org/10.1016/j.patter.2023.100804

**Record and text:** [Publisher](https://www.sciencedirect.com/science/article/pii/S2666389923001599); [open article](https://pmc.ncbi.nlm.nih.gov/articles/PMC10499856/).

**Reading status:** Publication record and abstract/taxonomy description verified. Full-text retrieval was inconsistent; complete the relevant methodology reading before citing specific taxonomy categories.

**Use:** Prediction setting, unavailable features and the interpretation of model scores in Sections 1, 4 and 5.

**Limit:** Supports concern about leakage; it does not prove a particular problem is present in this dataset or certify our pipeline. **Recent-paper quota: yes.**

### R04 Evaluation under imbalance

Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE, 10*(3), Article e0118432. https://doi.org/10.1371/journal.pone.0118432

**Record and text:** [Publisher article](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432).

**Reading status:** Abstract and evaluation discussion inspected.

**Use:** Explain why precision-recall behaviour matters when positive outcomes are uncommon, in Sections 1, 5 and 9.

**Limit:** Motivates precision-recall analysis; it does not imply ROC AUC is always invalid or that average precision and every precision-recall integration rule are identical. **Recent-paper quota: no.**

### R05 Hyperparameter optimisation

Bischl, B., Binder, M., Lang, M., Pielok, T., Richter, J., Coors, S., Thomas, J., Ullmann, T., Becker, M., Boulesteix, A.-L., Deng, D., & Lindauer, M. (2023). Hyperparameter optimization: Foundations, algorithms, best practices, and open challenges. *WIREs Data Mining and Knowledge Discovery, 13*(2), Article e1484. https://doi.org/10.1002/widm.1484

**Record and text:** [Publisher full text](https://wires.onlinelibrary.wiley.com/doi/full/10.1002/widm.1484); [institutional copy](https://epub.ub.uni-muenchen.de/108819/).

**Reading status:** Publication details and overview inspected; consult the search-space and evaluation sections when implementing Part F.

**Use:** Selection of search method, search budget, parameter ranges and interpretation of selected scores in Sections 5 and 7.

**Limit:** A search framework does not guarantee performance improvement. Report the actual results under the same evaluation conditions. **Recent-paper quota: yes.**

## Sources shortlisted for the remaining sections

### R06 Tabular model benchmarking

Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, & A. Oh (Eds.), *Advances in neural information processing systems* (Vol. 35, pp. 507-520). Curran Associates. https://doi.org/10.52202/068431-0037

**Record and paper:** [Official proceedings](https://papers.nips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract.html).

**Reading status:** Official abstract and citation export checked.

**Use:** Context for comparing practical tabular models and accounting for tuning and data properties in Section 6.

**Limit:** Its strong tree-based results concern benchmarked methods including ensembles. They do not establish that our single decision tree will outperform logistic regression or KNN. **Recent-paper quota: yes.**

### R07 Algorithm choice and dataset properties

McElfresh, D., Khandagale, S., Valverde, J., Prasad C, V., Ramakrishnan, G., Goldblum, M., & White, C. (2023). When do neural nets outperform boosted trees on tabular data? In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, & S. Levine (Eds.), *Advances in neural information processing systems* (Vol. 36, pp. 76336-76369). Curran Associates. https://doi.org/10.52202/075280-3337

**Record and paper:** [Official proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/f06d5ebd4ff40b40dd97e30cee632123-Abstract-Datasets_and_Benchmarks.html).

**Reading status:** Abstract, introduction, benchmark design and official citation export checked. Use the seven-author proceedings version; online preprint records show a different author list.

**Use:** Motivate measured comparisons and tuning rather than assuming one model family always wins, in Sections 6 and 7.

**Limit:** A benchmark's average rankings are not predictions of this project's result. **Recent-paper quota: yes.**

### R08 Conventional and deep tabular models

Shwartz-Ziv, R., & Armon, A. (2022). Tabular data: Deep learning is not all you need. *Information Fusion, 81*, 84-90. https://doi.org/10.1016/j.inffus.2021.11.011

**Record and paper:** [Publisher record](https://www.sciencedirect.com/science/article/pii/S1566253521002360); [author's publication page](https://www.ravid-shwartz-ziv.com/publication/tabular/); [author preprint](https://arxiv.org/abs/2106.03253).

**Reading status:** Publication details and abstract inspected; targeted full-method reading remains.

**Use:** Additional context for choosing manageable tabular baselines in Section 6 if the final discussion needs it.

**Limit:** The study's methods and data differ from ours. Avoid using several benchmark citations to repeat the same claim or pad the reference list. Use the 2022 journal version rather than counting the 2021 preprint separately. **Recent-paper quota: yes.**

### R09 Ensembles and augmentation for imbalanced data

Khan, A. A., Chaudhari, O., & Chandra, R. (2024). A review of ensemble learning and data augmentation models for class imbalanced problems: Combination, implementation and evaluation. *Expert Systems with Applications, 244*, Article 122778. https://doi.org/10.1016/j.eswa.2023.122778

**Record and paper:** [Publisher record](https://doi.org/10.1016/j.eswa.2023.122778); [author manuscript](https://arxiv.org/abs/2304.02858).

**Reading status:** Journal publication metadata, abstract and author manuscript abstract checked. The journal year is 2024 despite 2023 appearing in the DOI and preprint.

**Use:** Compare justified imbalance treatments and the interaction of augmentation with ensembles in Sections 4 and 8.

**Limit:** Reported improvements apply to the evaluated combinations and datasets. They do not require SMOTE here or establish a benefit for our ensemble. Check the final tables before citing exact counts of compared methods. **Recent-paper quota: yes.**

### R10 Ensemble construction

Sagi, O., & Rokach, L. (2018). Ensemble learning: A survey. *WIREs Data Mining and Knowledge Discovery, 8*(4), Article e1249. https://doi.org/10.1002/widm.1249

**Record and text:** [Publisher](https://wires.onlinelibrary.wiley.com/doi/10.1002/widm.1249).

**Reading status:** Publication details and abstract inspected. Read the relevant voting or stacking material when the implemented method is chosen.

**Use:** Motivate combining predictions and discuss construction choices in Section 8.

**Limit:** General ensemble motivation does not establish improvement for this pair. Measure effectiveness and added complexity against the members. **Recent-paper quota: no.**

## Separate dataset and implementation sources

Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing* [Data set]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306

The journal article and dataset have different author ordering in their supplied citations. Preserve the appropriate order for each. Use `bank-additional-full.csv` and its own names document; the landing page combines information from multiple variants.

Use official scikit-learn documentation for implementation details, including pipelines, time-ordered splits, scoring functions and model parameters. Record the installed version with experiment outputs. Documentation supports implementation claims but is not one of the ten academic papers above.

## Reference use and verification checks

Only sources actually cited should appear in the final report bibliography. Add sources if an important technical claim remains unsupported; replace redundant sources rather than forcing them into the argument. Confirm dates and the recent-paper count against the final submission requirements. Recheck publication notices before final submission, and keep the inspected version clear when preprint and publication details differ.

A screened 2023 healthcare metrics review was excluded after an expression-of-concern record appeared in the search. It is not included in the ten-paper shortlist or used to support the draft.

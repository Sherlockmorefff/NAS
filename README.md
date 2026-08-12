D-VAE -- A Variational Autoencoder for Directed Acyclic Graphs
===============================================================================

The optional Exact-GP clustered initializers are documented in
[docs/wgmm_clustered_ted.md](docs/wgmm_clustered_ted.md) and
[docs/gmm_clustered_schur.md](docs/gmm_clustered_schur.md). The latter covers
the fixed K=6 ordinary diagonal-GMM plus shared-kernel conditional Schur plus
low-fidelity path. The default Phase4 initializer remains the legacy Schur
strategy.

The unified node-classification dataset protocol and isolated S0/G100/G150
runner are documented in
[docs/cross_dataset_protocol.md](docs/cross_dataset_protocol.md).

Repository layout and launch scripts
------------------------------------

The current code, experiment-state, and artifact responsibilities are described
in [docs/repository_layout.md](docs/repository_layout.md). Formal Python entry
points such as `bo_phase4.py`, `cross_dataset_runner.py`, and `final_eval.py`
remain at the repository root.

Experiment artifact classification and retention rules are documented in
[docs/experiment_artifact_guide.md](docs/experiment_artifact_guide.md).
New formal runs use a fixed `protocol_id` and write only below
`results/{search,final_eval,posthoc}/<protocol_id>/` and
`logs/<protocol_id>/`. Pre-2026-08-13 outputs are local historical inputs below
`legacy_artifacts/pre_20260813/`; ignored artifacts are not disposable caches.

Maintained shell launchers are organized under `scripts/`:

    scripts/analysis/run_collect_results.sh --help
    scripts/analysis/run_diagnostics_suite.sh --help
    scripts/evaluation/run_final_eval_topk_seedfair.sh --help

The historical paths remain compatible wrappers, so existing commands using
`analyse/run_collect_results.sh`, `analyse/run_diagnostics_suite.sh`, or the
root `run_final_eval_topk_seedfair.sh` continue to work.

Validation and preflight implementations live under `scripts/validation/`;
their former root commands remain compatibility modules. Repository artifact
inventories can be generated without reading result contents:

    python scripts/validation/resource_preflight.py --help
    python scripts/maintenance/inventory_repository.py \
      --repo-root "$PWD" \
      --output-dir artifacts/indexes/repository_inventory_<timestamp>

Only `tests/` is part of the default maintained pytest collection. Historical
programs under `legacy/` and vendored third-party tests are retained for
reproduction but are not implicitly collected.

About
-----

Directed acyclic graphs (DAGs) are of particular interest to machine learning researchers, as many machine learning models are realized as computations on DAGs, including neural networks and Bayesian networks. Two important problems, neural architecture search (NAS) and Bayesian network structure learning (BNSL), are essentially DAG optimization problems, where an optimal DAG structure is to be found to best fit a given dataset.

D-VAE is a variational autoencoder for DAGs. It encodes/decodes DAGs using an asynchronous message passing scheme where a node updates its state only after all its predecessors' have been updated. The final node's state can injectively encode the computation on a DAG, rather than only encoding local structures as in standard simultaneous message passing. After training on some DAG distribution, D-VAE can not only generate novel and valid DAGs, but also be used to optimize DAG structures in its latent space. By embedding DAGs into a continuous latent space, D-VAE transforms the difficult discrete optimization problem into an easier continuous space optimization problem, where principled Bayesian optimization can be performed in this latent space to optimize DAG structures. Thanks to the computation-encoding property, D-VAE also empirically embeds DAGs with similar computation purposes (and performances) into the same region, which greatly facilitates the Bayesian optimization.

For more information, please check our paper:
> M. Zhang, S. Jiang, Z. Cui, R. Garnett, Y. Chen, D-VAE: A Variational Autoencoder for Directed Acyclic Graphs, Advances in Neural Information Processing Systems (NeurIPS-19). [\[PDF\]](https://arxiv.org/pdf/1904.11088.pdf)

Installation
------------

Tested with Python 3.6, PyTorch 0.4.1.

Install [PyTorch](https://pytorch.org/) >= 0.4.1

Install python-igraph by:

    pip install python-igraph

Install pygraphviz by:

    conda install graphviz
    conda install pygraphviz

Other required python libraries: tqdm, six, scipy, numpy, matplotlib

Training
--------

### Neural Architectures

    python train.py --data-name final_structures6 --save-interval 100 --save-appendix _DVAE --epochs 300 --lr 1e-4 --model DVAE --bidirectional --nz 56 --batch-size 32

### Bayesian Networks

    python train.py --data-name asia_200k --data-type BN --nvt 8 --save-interval 50 --save-appendix _DVAE --epochs 100 --lr 1e-4 --model DVAE_BN --nz 56 --batch-size 128

Bayesian Optimization
---------------------

To perform Bayesian optimization experiments after training D-VAE, the following additional steps are needed.

Install sparse Gaussian Process (SGP) based on Theano:

    cd bayesian_optimization/Theano-master/
    python setup.py install
    cd ../..

Download the [CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) by: 

    cd software/enas
    mkdir data
    cd data
    wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    tar -xzvf cifar-10-python.tar.gz
    mv cifar-10-batches-py/ cifar10/
    cd ../..

Download the 6-layer [pretrained ENAS model](https://drive.google.com/drive/folders/1e-mYRZS_10Aegj8Sczcb948RbHyiju1S?usp=sharing) to "software/enas/" (for evaluating a neural architecture's weight-sharing accuracy). There should be a folder named "software/enas/outputs_6/", which contains four model files. The 12-layer pretrained ENAS model is available [here](https://drive.google.com/drive/folders/18GU9g5DNiHn2MOVKOiF1fCwNQMTA-mnH?usp=sharing) too.

Install [TensorFlow](https://www.tensorflow.org/install/gpu) >= 1.12.0

Install R package _bnlearn_:

    R
    install.packages('bnlearn', lib='/R/library', repos='http://cran.us.r-project.org')

Then, in "bayesian_optimization/", type:

    ./run_bo_ENAS.sh

and 

    ./run_bo_BN.sh

to run Bayesian optimization for neural architecturs and Bayesian networks, respectively.

Finally, to summarize the BO results, type:

    python summarize.py

The results will be saved in "bayesian_optimization/**_aggregate_results/". The settings can be changed within "summarize.py".

Reference
---------

If you find the code useful, please cite our paper.

    @article{zhang2019d,
      title={D-VAE: A Variational Autoencoder for Directed Acyclic Graphs},
      author={Zhang, Muhan and Jiang, Shali and Cui, Zhicheng and Garnett, Roman and Chen, Yixin},
      booktitle={Advances in Neural Information Processing Systems},
      pages={1586--1598},
      year={2019}
    } 

Muhan Zhang, Washington University in St. Louis
muhan@wustl.edu
5/13/2019

# vaeda

vaeda (variaitonal auto-encoder (vae) for doublet annotation (da)) is a python package for doublet annotation in single cell RNA-sequencing. For method details and comparisons to alternative doublet annotation tools, see our [pre-print](https://biorxiv.org/cgi/content/short/2022.04.15.488440v1).

#### Installation

You can install vaeda using pip as follows:

```
conda create -n vaeda_env python=3.8
conda activate vaeda_env

conda install -c conda-forge tensorflow=2.4.0
pip install --upgrade tensorflow-probability==0.12.2
conda install -c bioconda scanpy=1.7.2
pip install typing-extensions==3.7.4 absl-py==0.10 six==1.15.0 wrapt==1.12.1 xlrd==1.2.0
pip install -i https://test.pypi.org/simple/ vaeda==0.0.17
```

#### Quick Start

In the following, X is a numpy array holding raw counts with cells as rows and genes as columns.

```
import vaeda

preds, preds_on_P, calls, encoding, knn_feature = vaeda.vaeda(X)

```

Where:
* preds are the doublet scores on the input data X
* preds_on_P are doublet scores on the simulated doublets
* calls are the doublet calls (ie, doublet or singlet) on the input data X
* encoding is the low dimensional latent representation of the input data learned by vaeda's variational auto-encoder
* knn_feature are the preliminary knn scores on the input data X

#### Other doublet detection tools:

* [scds](https://github.com/kostkalab/scds)
* [scDblFInder](https://github.com/plger/scDblFinder)
* [DoubletFinder](https://github.com/chris-mcginnis-ucsf/DoubletFinder)
* [Scrublet](https://github.com/AllonKleinLab/scrublet)
* [solo](https://github.com/calico/Solo)



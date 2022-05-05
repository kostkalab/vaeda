import setuptools
  
with open("README.md", "r") as fh:
    description = fh.read()

setuptools.setup(
    name="vaeda",
    version="0.0.24",
    author="Hannah Schriever",
    author_email="hcs31@pitt.edu",
    packages=["vaeda"],
    description="A computational tool for annotating doublets in scRNAseq data.",
    long_description=description,
    long_description_content_type="text/markdown",
    url="https://github.com/kostkalab/vaeda",
    license='MIT',
    python_requires='>=3.8',
    install_requires=['numpy', 'tensorflow', 'scipy', 'scikit-learn', 'kneed', 'anndata', 'tensorflow_probability', 'scanpy>1.3.3']
)


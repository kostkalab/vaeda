import setuptools
  
with open("README.md", "r") as fh:
    description = fh.read()

setuptools.setup(
    name="vaeda",
    version="0.0.1",
    author="Hannah Schriever",
    author_email="hcs31@pitt.edu",
    packages=["vaeda"],
    description="A computational tool for annotating doublets in scRNAseq data.",
    long_description=description,
    long_description_content_type="text/markdown",
    url="https://github.com/kostkalab/vaeda",
    license='MIT',
    python_requires='>=3.8',
    install_requires=['numpy', 'tensorflow', 'scipy', 'pathlib', 'pandas', 'matplotlib', 'random', 'sklearn', 'kneed', 'scanpy', 'anndata', 'math']
)


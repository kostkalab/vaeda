import anndata
import numpy as np
import scanpy as sc
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA


def cluster(X, comp=20):
    adata = anndata.AnnData(X=X)
    adata.var_names_make_unique()

    sc.tl.pca(adata, svd_solver="arpack")
    n = int(np.sqrt(X.shape[0]))

    sc.pp.neighbors(adata, n_neighbors=n, n_pcs=comp)
    sc.tl.leiden(adata)

    clust = np.array(adata.obs["leiden"]).astype(int)

    return clust


def fast_cluster(X, comp=20):
    pca = PCA(n_components=comp, random_state=42)
    pca_proj = pca.fit_transform(X)

    n_clusters = int(X.shape[0] * 0.1)

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=0,
        batch_size=1024,
        max_iter=20,
    ).fit(pca_proj)

    mini_clust = kmeans.predict(pca_proj)

    meta_cells = np.zeros((mini_clust.max() + 1, X.shape[1]))
    for c in np.unique(mini_clust):
        meta_cells[c] = np.mean(X[mini_clust == c, :], axis=0)

    adata = anndata.AnnData(X=meta_cells)
    adata.var_names_make_unique()

    sc.tl.pca(adata, svd_solver="arpack", random_state=0)
    n = int(np.sqrt(meta_cells.shape[0]))

    sc.pp.neighbors(adata, n_neighbors=n, n_pcs=comp, random_state=0)
    sc.tl.leiden(adata, random_state=0)

    clust_meta = np.array(adata.obs["leiden"]).astype(int)

    clust = np.zeros(X.shape[0])
    for c in np.unique(clust_meta):
        meta_ind = np.where(clust_meta == c)
        ind = np.isin(mini_clust, meta_ind)

        clust[ind] = c

    return clust

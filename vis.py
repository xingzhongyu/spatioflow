import scanpy as sc
import numpy as np
day=5
adata = sc.read_h5ad(f"data/day{day}_pred.h5ad")
print(adata)
origin_adata=sc.read_h5ad(f"data/day{day}.h5ad")
print(origin_adata)
# 从origin_adata的annotation类型中随机赋值
unique_annotations = origin_adata.obs['Annotation'].unique()
adata.obs['Annotation'] = np.random.choice(unique_annotations, size=len(adata), replace=True)
# sc.pl.spatial expects the coordinate key via `basis` (default "spatial")
sc.pl.spatial(adata, basis="spatial", save=f"day{day}_pred.png", spot_size=20, color='Annotation',palette=['blue'], 
               alpha=1.0, frameon=False)
sc.pl.spatial(origin_adata, basis="spatial", save=f"day{day}.png", spot_size=20, color='Annotation',palette=['blue'], 
               alpha=1.0, frameon=False)


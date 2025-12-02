import re

import scanpy as sc


def main():
    adata = sc.read_h5ad("Regeneration.h5ad")
    adata = adata[
        adata.obs["Batch"].isin(
            [
                "Injury_2DPI_rep1_SS200000147BL_D5",
                "Injury_5DPI_rep1_SS200000147BL_D2",
                "Injury_10DPI_rep1_SS200000147BL_B5",
                "Injury_15DPI_rep4_FP200000266TR_E4",
                "Injury_20DPI_rep2_SS200000147BL_B4",
            ]
        )
    ]
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
    adata.write_h5ad("Regeneration_2000.h5ad")

    adata = sc.read_h5ad("Regeneration_2000.h5ad")

    for batch in adata.obs["Batch"].unique():
        adata_batch = adata[adata.obs["Batch"] == batch]
        pattern = r"Injury_(\d+)DPI"
        day_match = re.search(pattern, batch)
        if day_match is None:
            continue
        day_label = day_match.group(1)
        adata_batch.write_h5ad(f"day{day_label}.h5ad")


if __name__ == "__main__":
    main()


# GitHub Upload Steps

This repository has already been uploaded to:

https://github.com/kwanxiang/serial-cxr-edema-reproducibility

Original upload checklist:

1. Create a new public GitHub repository.
2. Replace `https://github.com/USERNAME/REPOSITORY` in `CITATION.cff` and `docs/DATA_AVAILABILITY_STATEMENT.md`.
3. Confirm that no raw medical images, downloaded source data, credentials, or local scratch files were added.

Command-line upload example:

```bash
cd reproducibility_package
git init
git add .
git commit -m "Initial reproducibility package"
git branch -M main
git remote add origin https://github.com/USERNAME/REPOSITORY.git
git push -u origin main
```

After the repository is public, update the manuscript declarations with the final GitHub URL. If you create a Zenodo DOI from the GitHub release, use the DOI in the manuscript instead of the GitHub URL.

# lietorch Vendor Notes

- Source: https://github.com/princeton-vl/lietorch.git
- Commit: e7df86554156b36846008d8ddbcc4d8521a16554
- License: BSD-3-Clause; upstream license file is copied under `upstream/`.
- Vendored subset: package source, CUDA/C++ extension source, setup metadata, README, and license.

`upstream/eigen` is a symlink to `../../eigen/upstream` so lietorch's upstream
`setup.py` can build against the project-local Eigen dependency without carrying
a second Eigen copy.

Local changes: none.

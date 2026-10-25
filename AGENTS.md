# Project output conventions

- Save all analysis plots as PNG files under the project `plots/` directory.
- Keep `plots/` image-only. Save plotted tables and JSON provenance in `results/plot_data/`, and migration or housekeeping records in `results/maintenance/`.
- Update plotting scripts and document links to preserve this convention.
- Compiled document PDFs belong beside their LaTeX sources; they are not plot exports.

# Reviewable work and checkpoints

- Keep scientific, diagnostic, and debugging scripts in the repository, never
  only in `/tmp`. Preserve the inputs and findings needed to review them later.
- Commit and push coherent milestones, including before long computations.
  Keep large generated results under the existing ignored output directories.

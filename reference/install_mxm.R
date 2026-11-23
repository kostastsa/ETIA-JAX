# Install MXM 1.5.8 (archived on CRAN on 2026-11-03) and its dependencies
# into ~/R/library. ETIA's AFS runs FBED and SES with MXM; the increment-8
# fixtures need it. Run:
#
#     Rscript reference/install_mxm.R
#
lib <- path.expand("~/R/library")
dir.create(lib, showWarnings = FALSE, recursive = TRUE)
.libPaths(c(lib, .libPaths()))
options(repos = c(CRAN = "https://cloud.r-project.org"), Ncpus = 16)

tarball <- file.path(tempdir(), "MXM_1.5.8.tar.gz")
download.file("https://cran.r-project.org/src/contrib/Archive/MXM/MXM_1.5.8.tar.gz", tarball)
untar(tarball, exdir = tempdir())
desc <- read.dcf(file.path(tempdir(), "MXM", "DESCRIPTION"))
fields <- intersect(c("Depends", "Imports", "LinkingTo"), colnames(desc))
deps <- unlist(strsplit(gsub("\\([^)]*\\)", "", paste(desc[, fields], collapse = ",")), "[,\n ]+"))
deps <- setdiff(trimws(deps), c("", "R", rownames(installed.packages())))
cat("dependencies to install:", deps, "\n")
if (length(deps)) install.packages(deps, lib = lib)
install.packages(tarball, repos = NULL, type = "source", lib = lib)
library(MXM, lib.loc = lib)
cat("MXM", as.character(packageVersion("MXM")), "installed in", lib, "\n")

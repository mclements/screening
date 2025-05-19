#' Code to use the screening package inline. Not directly called by the user.
#' @param ... arguments
#' @keywords internal
#' @rdname plugin
inlineCxxPlugin <- function(...) {
    ismacos <- Sys.info()[["sysname"]] == "Darwin"
    openmpflag <- if (ismacos) "" else "$(SHLIB_OPENMP_CFLAGS)"
    plugin <- Rcpp::Rcpp.plugin.maker(include.before = '#include "screening.h"',
                                      libs = paste(openmpflag,
                                                   "$(LAPACK_LIBS) $(BLAS_LIBS) $(FLIBS)"),
                                      package = "screening")
    settings <- plugin()
    settings$env$PKG_CPPFLAGS <- paste("-I../inst/include", openmpflag)
    ## if (!ismacos) settings$env$USE_CXX11 <- "yes"
    settings
}

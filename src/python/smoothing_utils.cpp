#include <armadillo>

// pybind11
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

// Carma
#include <carma/carma.h>

struct _neigh {
    arma::mat data;
    arma::colvec weights;
    arma::uword n_rows;
    _neigh(const arma::mat& m, const arma::mat& w):
        data(w.n_elem, m.n_cols, arma::fill::zeros),
        weights(w.n_elem, arma::fill::zeros),
        n_rows(0) {}
};

typedef _neigh neigh_t;

void neigh_vec(neigh_t& n,
               const arma::mat& m,
               const arma::uword m_nrow,
               const arma::uword m_ncol,
               const arma::mat& w,
               const arma::uword m_b,
               const arma::uword m_i,
               const arma::uword m_j) {

    arma::uword w_leg_i = w.n_rows / 2, w_leg_j = w.n_cols / 2;

    // copy values
    arma::uword k = 0;
    for (arma::uword i = 0; i < w.n_rows; ++i)
        for (arma::uword j = 0; j < w.n_cols; ++j)
            if (m_i + i >= w_leg_i && m_j + j >= w_leg_j &&
                m_i + i < w_leg_i + m_nrow &&
                m_j + j < w_leg_j + m_ncol &&
                arma::is_finite(m(m_j + m_i * m_ncol, 0))) {

                n.data(k, m_b) = m((m_j + j - w_leg_j) +
                    (m_i + i - w_leg_i) * m_ncol, m_b);
                n.weights(k++) = w(i, j);
            }
    n.n_rows = k;
}

arma::colvec nm_post_mean_x(const arma::colvec& x,
                            const arma::mat& sigma,
                            const arma::colvec& mu0,
                            const arma::mat& sigma0) {

    // inverse sigma0
    arma::mat inv_sum_weights(arma::size(sigma0));
    inv_sum_weights = arma::inv(sigma + sigma0);

    return sigma * inv_sum_weights * mu0 + sigma0 * inv_sum_weights * x;
}

arma::mat bayes_smoother(const arma::mat& m,
                         const arma::uword m_nrow,
                         const arma::uword m_ncol,
                         const arma::mat& w,
                         const arma::mat& sigma,
                         bool covar_sigma0) {

    // initialize result matrix
    arma::mat res(arma::size(m), arma::fill::none);
    res.fill(arma::datum::nan);

    // prior mean vector (neighbourhood)
    arma::colvec mu0(m.n_cols, arma::fill::zeros);

    // prior co-variance matrix (neighbourhood)
    arma::mat sigma0(arma::size(sigma), arma::fill::zeros);

    // neighbourhood
    neigh_t neigh(m, w);

    // compute values for each pixel
    for (arma::uword i = 0; i < m_nrow; ++i)
        for (arma::uword j = 0; j < m_ncol; ++j) {

            // fill neighbours values
            for (arma::uword b = 0; b < m.n_cols; ++b)
                neigh_vec(neigh, m, m_nrow, m_ncol, w, b, i, j);

            if (neigh.n_rows == 0) continue;

            // compute prior mean
            mu0 = arma::mean(neigh.data.rows(0, neigh.n_rows - 1), 0).as_col();

            // compute prior sigma
            sigma0 = arma::cov(neigh.data.rows(0, neigh.n_rows - 1), 0);

            // prior sigma covariance
            if (!covar_sigma0) {

                // clear non main diagonal cells
                sigma0.elem(arma::trimatu_ind(
                        arma::size(sigma0), 1)).fill(0.0);
                sigma0.elem(arma::trimatl_ind(
                        arma::size(sigma0), -1)).fill(0.0);
            }

            // evaluate multivariate bayesian
            res.row(j + i * m_ncol) =
                nm_post_mean_x(m.row(j + i * m_ncol).as_col(),
                               sigma, mu0, sigma0).as_row();
        }
    return res;
}

arma::mat kernel_smoother(const arma::mat& m,
                          const arma::uword m_nrow,
                          const arma::uword m_ncol,
                          const arma::mat& w,
                          const bool normalised) {

    // initialize result matrix
    arma::mat res(arma::size(m), arma::fill::none);
    res.fill(arma::datum::nan);

    // neighbourhood
    neigh_t neigh(m, w);

    // compute values for each pixel
    for (arma::uword b = 0; b < m.n_cols; ++b)
        for (arma::uword i = 0; i < m_nrow; ++i)
            for (arma::uword j = 0; j < m_ncol; ++j) {

                // fill neighbours values
                neigh_vec(neigh, m, m_nrow, m_ncol, w, b, i, j);

                if (neigh.n_rows == 0) continue;

                // normalise weight values
                if (normalised)
                    neigh.weights = neigh.weights /
                        arma::sum(neigh.weights.subvec(0, neigh.n_rows - 1));

                // compute kernel neighbourhood weighted mean
                res(j + i * m_ncol, b) = arma::as_scalar(
                    neigh.weights.subvec(0, neigh.n_rows - 1).as_row() *
                        neigh.data.col(b).subvec(0, neigh.n_rows - 1));
            }
    return res;
}

arma::mat bilinear_smoother(const arma::mat& m,
                            const arma::uword m_nrow,
                            const arma::uword m_ncol,
                            const arma::mat& w,
                            double tau) {

    // initialize result matrix
    arma::mat res(arma::size(m), arma::fill::none);
    res.fill(arma::datum::nan);

    // neighbourhood
    neigh_t neigh(m, w);

    // compute values for each pixel
    for (arma::uword b = 0; b < m.n_cols; ++b)
        for (arma::uword i = 0; i < m_nrow; ++i)
            for (arma::uword j = 0; j < m_ncol; ++j) {

                // fill neighbours values
                neigh_vec(neigh, m, m_nrow, m_ncol, w, b, i, j);

                if (neigh.n_rows == 0) continue;

                // compute bilinear weight
                arma::colvec bln_weight = neigh.weights % arma::normpdf(
                    neigh.data.col(b) - m(j + i * m_ncol, b), 0, tau);

                // normalise weight values
                bln_weight = bln_weight /
                    arma::sum(bln_weight.subvec(0, neigh.n_rows - 1));

                // compute kernel neighbourhood weighted mean
                res(j + i * m_ncol, b) = arma::as_scalar(
                    bln_weight.subvec(0, neigh.n_rows - 1).as_row() *
                    neigh.data.col(b).subvec(0, neigh.n_rows - 1));
            }
    return res;
}

void PyInit_smoothing(py::module &m) {
    m.def("bayes_smoother", &bayes_smoother, "Bayes Smoother");

    m.def("kernel_smoother", &kernel_smoother, "Kernel Smoother");

    m.def("bilinear_smoother", &bilinear_smoother, "Bilinear Smoother");
}

#include <pybind11/pybind11.h>

namespace py = pybind11;

void PyInit_smoothing(py::module &);

PYBIND11_MODULE(smoothing, m) {
    // Optional docstring
    m.doc() = "Spatial Smoothing Functions";
    
    PyInit_smoothing(m);
}

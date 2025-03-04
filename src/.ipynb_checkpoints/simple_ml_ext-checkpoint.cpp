#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>
#include <vector>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    size_t now_sample = 0;
	auto Z = std::vector<std::vector<float>>(batch, std::vector<float>(k, 0.0));
	
	while (now_sample < m) {
		if (now_sample + batch > m) {
			// last batch process
			batch = m - now_sample;
		}
		
		// compute Z (after normalize)
		// X@theta  m*n n*k
		// i: row of X
		// j: column of theta
		// o: iteration of n
		for (size_t i = 0 ; i < batch ; ++i){
			float z_row_sum = 0.0;
			for (size_t j = 0 ; j < k ; ++j){
				Z[i][j] = 0.0;
				for (size_t o = 0 ; o < n ; ++o) {
					Z[i][j] += X[(now_sample + i)*n + o] * theta[o*k+j];
				}
				Z[i][j] = std::exp(Z[i][j]);
				z_row_sum += Z[i][j];
			}
			for (size_t j = 0 ; j < k ; ++j){
				Z[i][j] /= z_row_sum;
			}
		}
		
		// z = z - Iy
		for (size_t i = 0 ; i < batch ; ++i){
			Z[i][y[now_sample + i]] -= 1.0;
		}
		
		// grad = tX.T @ z  n*m  @  m*k
		for (size_t i = 0 ; i < n ; ++i) {
			for (size_t j = 0 ; j < k ; ++j) {
				float diff = 0.0;
				for (size_t o = 0 ; o < batch ; ++o) {
					// i: row of tx.T, column of tX
					// o: column of tx.T, row of tX
					diff += X[(now_sample + o) * n + i] * Z[o][j];
				}
				theta[i*k+j] -= lr* diff / batch;
			}
		}
		
		now_sample += batch;
	}
	
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
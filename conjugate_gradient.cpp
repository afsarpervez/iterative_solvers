#include <iostream>
#include <vector>
#include <cmath>

// Function to perform matrix-vector multiplication
std::vector<double> matVecMultiply(const std::vector<std::vector<double>>& A, const std::vector<double>& x) {
    size_t n = x.size();
    std::vector<double> result(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            result[i] += A[i][j] * x[j];
        }
    }
    return result;
}

// Function to compute dot product of two vectors
double dotProduct(const std::vector<double>& a, const std::vector<double>& b) {
    double result = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result;
}

// Function to compute vector addition
std::vector<double> vectorAdd(const std::vector<double>& a, const std::vector<double>& b) {
    std::vector<double> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }
    return result;
}

// Function to compute vector subtraction
std::vector<double> vectorSubtract(const std::vector<double>& a, const std::vector<double>& b) {
    std::vector<double> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] - b[i];
    }
    return result;
}

// Function to compute scalar-vector multiplication
std::vector<double> scalarMultiply(const std::vector<double>& a, double scalar) {
    std::vector<double> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] * scalar;
    }
    return result;
}

// Conjugate Gradient method
std::vector<double> conjugateGradient(const std::vector<std::vector<double>>& A, const std::vector<double>& b, const std::vector<double>& x0, double tol) {
    size_t n = b.size();
    std::vector<double> x = x0;
    std::vector<double> r = vectorSubtract(b, matVecMultiply(A, x));
    std::vector<double> p = r;
    std::vector<double> r_old;
    double alpha, beta;
    int k = 0;

    while (dotProduct(r, r) > tol * tol) {
        std::vector<double> Ap = matVecMultiply(A, p);
        alpha = dotProduct(r, r) / dotProduct(p, Ap);
        x = vectorAdd(x, scalarMultiply(p, alpha));
        r_old = r;
        r = vectorSubtract(r, scalarMultiply(Ap, alpha));

        if (dotProduct(r, r) < tol * tol) {
            break;
        }

        beta = dotProduct(r, r) / dotProduct(r_old, r_old);
        p = vectorAdd(r, scalarMultiply(p, beta));
        k++;
    }

    return x;
}

int main() {
    // Define the symmetric positive-definite matrix A
    std::vector<std::vector<double>> A = {
        {4, 1},
        {1, 3}
    };

    // Define the right-hand side vector b
    std::vector<double> b = {1, 2};

    // Initial guess x0
    std::vector<double> x0 = {2, 1};

    // Tolerance
    double tol = 1e-6;

    // Solve Ax = b using Conjugate Gradient method
    std::vector<double> x = conjugateGradient(A, b, x0, tol);

    // Print the solution
    std::cout << "Solution: ";
    for (double xi : x) {
        std::cout << xi << " ";
    }
    std::cout << std::endl;

    return 0;
}

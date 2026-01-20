#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <cmath>
#include <string>

namespace py = pybind11;

/* ================= Scalar ================= */
class Scalar {
public:
    double v;

    Scalar(double v = 0.0) : v(v) {}

    Scalar operator+(const Scalar& o) const { return v + o.v; }
    Scalar operator-(const Scalar& o) const { return v - o.v; }
    Scalar operator*(const Scalar& o) const { return v * o.v; }
    Scalar operator/(const Scalar& o) const { return v / o.v; }

    Scalar operator+(double o) const { return v + o; }
    Scalar operator*(double o) const { return v * o; }

    double sin() const { return std::sin(v); }
    double cos() const { return std::cos(v); }
    double tan() const { return std::tan(v); }
    double asin() const { return std::asin(v); }
    double acos() const { return std::acos(v); }
    double atan() const { return std::atan(v); }
    double sinh() const { return std::sinh(v); }
    double cosh() const { return std::cosh(v); }
    double tanh() const { return std::tanh(v); }
    double exp() const { return std::exp(v); }
    double log() const { return std::log(v); }
    double log10() const { return std::log10(v); }
    double sqrt() const { return std::sqrt(v); }
    double cbrt() const { return std::cbrt(v); }
    double abs() const { return std::abs(v); }
    double pow(double n) const { return std::pow(v, n); }
};

/* ================= Vec3 ================= */
class Vec3 {
public:
    double x, y, z;

    Vec3(double x = 0, double y = 0, double z = 0)
        : x(x), y(y), z(z) {
    }

    Vec3 operator+(const Vec3& o) const {
        return { x + o.x, y + o.y, z + o.z };
    }

    Vec3 operator-(const Vec3& o) const {
        return { x - o.x, y - o.y, z - o.z };
    }

    Vec3 operator*(double s) const {
        return { x * s, y * s, z * s };
    }

    double dot(const Vec3& o) const {
        return x * o.x + y * o.y + z * o.z;
    }

    Vec3 cross(const Vec3& o) const {
        return {
            y * o.z - z * o.y,
            z * o.x - x * o.z,
            x * o.y - y * o.x
        };
    }

    double magnitude() const {
        return std::sqrt(dot(*this));
    }

    Vec3 normalize() const {
        double m = magnitude();
        return m == 0 ? Vec3() : (*this) * (1.0 / m);
    }

    double distance(const Vec3& o) const {
        return (*this - o).magnitude();
    }
};

/* ================= Solver ================= */
class Solver {
public:
    static double integrate(py::function f, double a, double b, int n) {
        if (n <= 0 || n % 2 != 0)
            throw std::runtime_error("n must be positive and even");

        double h = (b - a) / n;
        double s = f(a).cast<double>() + f(b).cast<double>();

        for (int i = 1; i < n; ++i) {
            double x = a + h * i;
            s += (i % 2 ? 4 : 2) * f(x).cast<double>();
        }
        return s * h / 3.0;
    }
};

/* ================= Constants ================= */
namespace Constants {
    constexpr double pi = 3.141592653589793;
    constexpr double e = 2.718281828459045;
    constexpr double c = 299792458.0;
    constexpr double h = 6.62607015e-34;
    constexpr double k = 1.380649e-23;
    constexpr double g = 9.80665;
} // <-- namespace properly closed

/* ================= PYBIND ================= */
PYBIND11_MODULE(calculus_core, m) {
    m.doc() = "CalCulus v0.3.0 C++ Scientific Engine";

    py::class_<Scalar>(m, "Scalar")
        .def(py::init<double>())
        .def_readwrite("value", &Scalar::v)
        .def("__repr__", [](const Scalar& s) {
        return "<Scalar " + std::to_string(s.v) + ">";
            })
        .def("__float__", [](const Scalar& s) { return s.v; })
        .def("__add__", [](const Scalar& a, const Scalar& b) { return a + b; })
        .def("__add__", [](const Scalar& a, double b) { return a + b; })
        .def("__radd__", [](const Scalar& a, double b) { return a + b; })
        .def("__sub__", [](const Scalar& a, const Scalar& b) { return a - b; })
        .def("__mul__", [](const Scalar& a, const Scalar& b) { return a * b; })
        .def("__mul__", [](const Scalar& a, double b) { return a * b; })
        .def("__rmul__", [](const Scalar& a, double b) { return a * b; })
        .def("__truediv__", [](const Scalar& a, const Scalar& b) { return a / b; })
        .def("sin", &Scalar::sin)
        .def("cos", &Scalar::cos)
        .def("tan", &Scalar::tan)
        .def("asin", &Scalar::asin)
        .def("acos", &Scalar::acos)
        .def("atan", &Scalar::atan)
        .def("sinh", &Scalar::sinh)
        .def("cosh", &Scalar::cosh)
        .def("tanh", &Scalar::tanh)
        .def("exp", &Scalar::exp)
        .def("log", &Scalar::log)
        .def("log10", &Scalar::log10)
        .def("sqrt", &Scalar::sqrt)
        .def("cbrt", &Scalar::cbrt)
        .def("abs", &Scalar::abs)
        .def("pow", &Scalar::pow);

    py::class_<Vec3>(m, "Vec3")
        .def(py::init<double, double, double>())
        .def("__add__", &Vec3::operator+)
        .def("__sub__", &Vec3::operator-)
        .def("__mul__", &Vec3::operator*)
        .def("dot", &Vec3::dot)
        .def("cross", &Vec3::cross)
        .def("magnitude", &Vec3::magnitude)
        .def("normalize", &Vec3::normalize)
        .def("distance", &Vec3::distance);

    py::class_<Solver>(m, "Solver")
        .def_static("integrate", &Solver::integrate);

    auto cst = m.def_submodule("Constants");
    cst.attr("pi") = Constants::pi;
    cst.attr("e") = Constants::e;
    cst.attr("c") = Constants::c;
    cst.attr("h") = Constants::h;
    cst.attr("k") = Constants::k;
    cst.attr("g") = Constants::g;
}

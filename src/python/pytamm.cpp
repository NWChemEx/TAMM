#include <tamm/op_executor.hpp>
#include <tamm/opmin.hpp>
#include <tamm/tamm.hpp>
#include <tamm/tamm_git.hpp>

#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;
using namespace tamm;

using LTd = tamm::LabeledTensor<double>;
using LTz = tamm::LabeledTensor<std::complex<double>>;

using NewOpBase = tamm::new_ops::Op;
using NewLTOp   = tamm::new_ops::LTOp;
using NewMultOp = tamm::new_ops::MultOp;
using NewAddOp  = tamm::new_ops::AddOp;

// For simplicity, we wrap Tensor, LabeledTensor, LocalTensor, and LabeledLocalTensor
// Directly binding the raw objects leads to significant issues

template<typename T>
struct PyTensor;
template<typename T>
struct PyLabeledTensor;
template<typename T>
struct PyLocalTensor;
template<typename T>
struct PyLabeledLocalTensor;

using PTd       = PyTensor<double>;
using PTz       = PyTensor<std::complex<double>>;
using PLTd      = PyLabeledTensor<double>;
using PLTz      = PyLabeledTensor<std::complex<double>>;
using PLocalTd  = PyLocalTensor<double>;
using PLocalTz  = PyLocalTensor<std::complex<double>>;
using PLLocalTd = PyLabeledLocalTensor<double>;
using PLLocalTz = PyLabeledLocalTensor<std::complex<double>>;

using PyScalar = std::variant<double, std::complex<double>>;

PYBIND11_MAKE_OPAQUE(SymbolTable);

// -----------------------------------------------------------------------------
// Conversion helpers
// These deal with types such as IndexVector and std::complex
// IndexVector is an alias of std::vector<uint32_t>
// As it is not recommended to directly bind std::vector, we let pybind11 implicitly convert Python
// iterables to C++ IndexVectors
// -----------------------------------------------------------------------------

static IndexVector to_index_vector(py::handle obj) {
  IndexVector out;
  for(auto item: py::reinterpret_borrow<py::iterable>(obj)) out.push_back(item.cast<Index>());
  return out;
}

static py::tuple from_index_vector(const IndexVector& vec) {
  py::tuple out(vec.size());
  for(size_t i = 0; i < vec.size(); ++i) out[i] = py::int_(vec[i]);
  return out;
}

template<typename T>
static py::object py_scalar_from_value(const T& value) {
  if constexpr(std::is_same_v<T, std::complex<double>>) return py::cast(value);
  else return py::float_(value);
}

static bool py_is_numeric_scalar(py::handle h) {
  return PyComplex_Check(h.ptr()) || py::isinstance<py::float_>(h) || py::isinstance<py::int_>(h);
}

static bool py_scalar_is_zero(py::handle h) {
  if(PyComplex_Check(h.ptr())) return h.cast<std::complex<double>>() == std::complex<double>{};
  if(py::isinstance<py::float_>(h)) return h.cast<double>() == 0.0;
  if(py::isinstance<py::int_>(h)) return h.cast<long long>() == 0;
  return false;
}

static PyScalar py_to_expr_scalar(py::handle h) {
  if(PyComplex_Check(h.ptr())) return h.cast<std::complex<double>>();
  if(py::isinstance<py::float_>(h)) return h.cast<double>();
  if(py::isinstance<py::int_>(h)) return static_cast<double>(h.cast<long long>());
  throw py::type_error("Expected a numeric scalar");
}

static tamm::Scalar py_to_tamm_scalar(py::handle h) {
  if(PyComplex_Check(h.ptr())) return tamm::Scalar{h.cast<std::complex<double>>()};
  if(py::isinstance<py::float_>(h)) return tamm::Scalar{h.cast<double>()};
  if(py::isinstance<py::int_>(h)) return tamm::Scalar{h.cast<int64_t>()};
  throw py::type_error("Expected a numeric scalar");
}

static py::object tamm_scalar_to_pyobject(const tamm::Scalar& s) {
  return std::visit([](const auto& v) -> py::object { return py::cast(v); }, s.value());
}

template<typename T>
static bool py_is_scalar_compatible(py::handle h) {
  if constexpr(std::is_same_v<T, double>) {
    return py::isinstance<py::float_>(h) || py::isinstance<py::int_>(h);
  }
  else {
    return PyComplex_Check(h.ptr()) || py::isinstance<py::float_>(h) || py::isinstance<py::int_>(h);
  }
}

template<typename T>
static T py_numeric_scalar_cast(py::handle h) {
  if constexpr(std::is_same_v<T, double>) {
    if(PyComplex_Check(h.ptr())) throw py::type_error("Expected a real numeric scalar");
    if(py::isinstance<py::float_>(h) || py::isinstance<py::int_>(h)) return h.cast<double>();
  }
  else {
    if(PyComplex_Check(h.ptr())) return h.cast<std::complex<double>>();
    if(py::isinstance<py::float_>(h)) return {h.cast<double>(), 0.0};
    if(py::isinstance<py::int_>(h)) return {static_cast<double>(h.cast<long long>()), 0.0};
  }
  throw py::type_error("Expected a numeric scalar");
}

static py::object numpy_array_protocol_result(py::array arr, py::object dtype = py::none(),
                                              py::object copy = py::none()) {
  if(dtype.is_none()) {
    if(!copy.is_none() && copy.cast<bool>()) return arr.attr("copy")();
    return arr;
  }

  const bool should_copy = !copy.is_none() && copy.cast<bool>();
  return arr.attr("astype")(dtype, py::arg("copy") = should_copy);
}

static void push_checked_dim(std::vector<size_t>& dims, std::vector<py::ssize_t>& shape, size_t n,
                             const char* where) {
  if(n > static_cast<size_t>(std::numeric_limits<py::ssize_t>::max())) {
    throw py::type_error(std::string(where) + ": dimension exceeds py::ssize_t");
  }
  dims.push_back(n);
  shape.push_back(static_cast<py::ssize_t>(n));
}

static size_t checked_product_or_one(const std::vector<size_t>& dims, const char* where) {
  size_t total = 1;
  for(size_t n: dims) {
    if(n != 0 && total > std::numeric_limits<size_t>::max() / n) {
      throw py::type_error(std::string(where) + ": tensor size overflows size_t");
    }
    total *= n;
  }
  return total;
}

// -----------------------------------------------------------------------------
// Strong numeric wrappers
// Dedicated class, as Python does not support type-safe operations to the degree required
// -----------------------------------------------------------------------------

template<typename StrongT, typename Underlying>
void bind_strong_num(py::module_& m, const char* name) {
  py::class_<StrongT>(m, name)
    .def(py::init<>())
    .def(py::init<Underlying>())
    .def("value", py::overload_cast<>(&StrongT::value, py::const_))
    .def("__int__", [](const StrongT& x) { return x.value(); })
    .def("__index__", [](const StrongT& x) { return x.value(); })
    .def("__repr__",
         [name](const StrongT& x) {
           return std::string(name) + "(" + std::to_string(x.value()) + ")";
         })
    .def("__hash__", [](const StrongT& x) { return std::hash<Underlying>{}(x.value()); })
    .def(py::self == py::self)
    .def(py::self != py::self)
    .def(py::self < py::self)
    .def(py::self <= py::self)
    .def(py::self > py::self)
    .def(py::self >= py::self)
    .def(py::self + py::self)
    .def(py::self - py::self)
    .def(py::self * py::self)
    .def(py::self / py::self)
    .def(py::self % py::self)
    .def(
      "__add__", [](const StrongT& a, Underlying b) { return a + b; }, py::is_operator())
    .def(
      "__sub__", [](const StrongT& a, Underlying b) { return a - b; }, py::is_operator())
    .def(
      "__mul__", [](const StrongT& a, Underlying b) { return a * b; }, py::is_operator())
    .def(
      "__truediv__", [](const StrongT& a, Underlying b) { return a / b; }, py::is_operator())
    .def(
      "__mod__", [](const StrongT& a, Underlying b) { return a % b; }, py::is_operator())
    .def(
      "__radd__", [](const StrongT& a, Underlying b) { return b + a; }, py::is_operator())
    .def(
      "__rsub__", [](const StrongT& a, Underlying b) { return b - a; }, py::is_operator())
    .def(
      "__rmul__", [](const StrongT& a, Underlying b) { return b * a; }, py::is_operator())
    .def(
      "__rtruediv__", [](const StrongT& a, Underlying b) { return b / a; }, py::is_operator())
    .def(
      "__rmod__", [](const StrongT& a, Underlying b) { return b % a; }, py::is_operator())
    .def(
      "__eq__", [](const StrongT& a, Underlying b) { return a == b; }, py::is_operator())
    .def(
      "__ne__", [](const StrongT& a, Underlying b) { return a != b; }, py::is_operator())
    .def(
      "__lt__", [](const StrongT& a, Underlying b) { return a < b; }, py::is_operator())
    .def(
      "__le__", [](const StrongT& a, Underlying b) { return a <= b; }, py::is_operator())
    .def(
      "__gt__", [](const StrongT& a, Underlying b) { return a > b; }, py::is_operator())
    .def(
      "__ge__", [](const StrongT& a, Underlying b) { return a >= b; }, py::is_operator())
    .def(
      "__iadd__",
      [](StrongT& a, Underlying b) -> StrongT& {
        a += b;
        return a;
      },
      py::is_operator())
    .def(
      "__isub__",
      [](StrongT& a, Underlying b) -> StrongT& {
        a -= b;
        return a;
      },
      py::is_operator())
    .def(
      "__imul__",
      [](StrongT& a, Underlying b) -> StrongT& {
        a *= b;
        return a;
      },
      py::is_operator())
    .def(
      "__itruediv__",
      [](StrongT& a, Underlying b) -> StrongT& {
        a /= b;
        return a;
      },
      py::is_operator());
}

// -----------------------------------------------------------------------------
// Tensor wrapper types
// -----------------------------------------------------------------------------

template<typename T>
static Tensor<T> make_tensor_from_sequence(py::sequence seq) {
  const auto n = py::len(seq);
  if(n == 0) throw py::value_error("Tensor construction from an empty sequence is ambiguous");

  bool all_tis = true;
  bool all_lbl = true;

  for(auto item: seq) {
    py::handle h = item;
    all_tis      = all_tis && py::isinstance<TiledIndexSpace>(h);
    all_lbl      = all_lbl && py::isinstance<TiledIndexLabel>(h);
  }

  if(all_tis) {
    TiledIndexSpaceVec spaces;
    spaces.reserve(static_cast<size_t>(n));
    for(auto item: seq) spaces.push_back(item.cast<TiledIndexSpace>());
    return Tensor<T>{spaces};
  }

  if(all_lbl) {
    IndexLabelVec labels;
    labels.reserve(static_cast<size_t>(n));
    for(auto item: seq) labels.push_back(item.cast<TiledIndexLabel>());
    return Tensor<T>{labels};
  }

  throw py::type_error("Tensor sequence constructor expects all TiledIndexSpace or all "
                       "TiledIndexLabel objects");
}

template<typename T>
struct PyTensor: std::enable_shared_from_this<PyTensor<T>> {
  using value_type = T;

  Tensor<T>                    tensor;
  std::shared_ptr<PyTensor<T>> view_parent;

  PyTensor() = default;
  explicit PyTensor(const Tensor<T>& t): tensor{t} {}
  explicit PyTensor(Tensor<T>&& t): tensor{std::move(t)} {}
  PyTensor(const Tensor<T>& t, std::shared_ptr<PyTensor<T>> parent):
    tensor{t}, view_parent{std::move(parent)} {}
  PyTensor(Tensor<T>&& t, std::shared_ptr<PyTensor<T>> parent):
    tensor{std::move(t)}, view_parent{std::move(parent)} {}
  explicit PyTensor(const TiledIndexSpaceVec& v): tensor{v} {}
  explicit PyTensor(const IndexLabelVec& v): tensor{v} {}

  Tensor<T>&       raw() { return tensor; }
  const Tensor<T>& raw() const { return tensor; }

  bool is_allocated() const { return tensor.is_allocated(); }

  void allocate(ExecutionContext& ec) {
    if(!tensor.is_allocated()) tensor.allocate(&ec);
  }

  void allocate_force(ExecutionContext& ec) { tensor.allocate(&ec); }

  void deallocate() {
    if(tensor.is_allocated()) tensor.deallocate();
  }

  bool same_memory_region(const PyTensor<T>& other) const {
    return tensor.memory_region() == other.tensor.memory_region();
  }

  bool same_distribution(const PyTensor<T>& other) const {
    return tensor.distribution() == other.tensor.distribution();
  }
};

template<typename T>
struct PyLabeledTensor {
  using value_type = T;

  std::shared_ptr<PyTensor<T>> tensor;
  IndexLabelVec                labels;

  PyLabeledTensor() = default;
  PyLabeledTensor(std::shared_ptr<PyTensor<T>> t, IndexLabelVec lbls):
    tensor{std::move(t)}, labels{std::move(lbls)} {}

  Tensor<T>& raw_tensor() const {
    if(!tensor) throw py::value_error("Invalid labeled tensor: missing tensor");
    return tensor->raw();
  }

  LabeledTensor<T> materialize() const {
    if(!tensor) throw py::value_error("Invalid labeled tensor: missing tensor");
    return LabeledTensor<T>{tensor->raw(), labels};
  }
};

template<typename T>
static std::shared_ptr<PyTensor<T>> wrap_tensor(const Tensor<T>& t) {
  return std::make_shared<PyTensor<T>>(t);
}

template<typename T>
static std::shared_ptr<PyLabeledTensor<T>>
make_labeled_tensor_from_args(std::shared_ptr<PyTensor<T>> self, py::args args) {
  if(!self) throw py::value_error("Tensor object is null");

  if(args.size() == 0) {
    auto lt = self->raw()();
    return std::make_shared<PyLabeledTensor<T>>(self, lt.labels());
  }

  bool all_lbl = true;
  bool all_str = true;
  for(auto a: args) {
    all_lbl = all_lbl && py::isinstance<TiledIndexLabel>(a);
    all_str = all_str && py::isinstance<py::str>(a);
  }

  if(all_lbl) {
    IndexLabelVec labels;
    labels.reserve(args.size());
    for(auto a: args) labels.push_back(a.cast<TiledIndexLabel>());
    return std::make_shared<PyLabeledTensor<T>>(self, std::move(labels));
  }

  if(all_str) {
    const auto& spaces = self->raw().tiled_index_spaces();
    if(args.size() != spaces.size()) {
      throw py::value_error("Number of string labels must match tensor rank");
    }

    IndexLabelVec labels;
    labels.reserve(args.size());
    for(size_t i = 0; i < args.size(); ++i) {
      labels.push_back(spaces[i].string_label(args[i].cast<std::string>()));
    }

    return std::make_shared<PyLabeledTensor<T>>(self, std::move(labels));
  }

  throw py::type_error("Tensor.__call__ expects all arguments to be strings or "
                       "TiledIndexLabel objects");
}

// -----------------------------------------------------------------------------
// Local tensor wrappers
// -----------------------------------------------------------------------------

// These manual methods are often useful for relatively ambiguous constructors
template<typename T>
static LocalTensor<T> make_local_tensor_from_tis_vec(const TiledIndexSpaceVec& spaces) {
  std::vector<TiledIndexSpace> v;
  v.reserve(spaces.size());
  for(const auto& tis: spaces) v.push_back(tis);
  return LocalTensor<T>(v);
}

template<typename T>
static LocalTensor<T> make_local_tensor_from_label_vec(const IndexLabelVec& labels) {
  std::vector<TiledIndexSpace> spaces;
  spaces.reserve(labels.size());
  for(const auto& lbl: labels) spaces.push_back(lbl.tiled_index_space());
  return LocalTensor<T>(spaces);
}

template<typename T>
static LocalTensor<T> make_local_tensor_from_size_vec(const std::vector<size_t>& sizes) {
  return LocalTensor<T>(sizes);
}

template<typename T>
struct PyLocalTensor: std::enable_shared_from_this<PyLocalTensor<T>> {
  using value_type = T;

  LocalTensor<T> tensor;

  PyLocalTensor() = default;
  explicit PyLocalTensor(const LocalTensor<T>& t): tensor{t} {}
  explicit PyLocalTensor(LocalTensor<T>&& t): tensor{std::move(t)} {}
  explicit PyLocalTensor(const TiledIndexSpaceVec& v):
    tensor{make_local_tensor_from_tis_vec<T>(v)} {}
  explicit PyLocalTensor(const IndexLabelVec& v): tensor{make_local_tensor_from_label_vec<T>(v)} {}
  explicit PyLocalTensor(const std::vector<size_t>& sizes):
    tensor{make_local_tensor_from_size_vec<T>(sizes)} {}

  LocalTensor<T>&       raw() { return tensor; }
  const LocalTensor<T>& raw() const { return tensor; }

  bool is_allocated() const { return tensor.is_allocated(); }

  void allocate(ExecutionContext& ec) {
    if(!tensor.is_allocated()) tensor.allocate(&ec);
  }

  void deallocate() {
    if(tensor.is_allocated()) tensor.deallocate();
  }
};

template<typename T>
struct PyLabeledLocalTensor {
  using value_type = T;

  std::shared_ptr<PyLocalTensor<T>> tensor;
  IndexLabelVec                     labels;

  PyLabeledLocalTensor() = default;
  PyLabeledLocalTensor(std::shared_ptr<PyLocalTensor<T>> t, IndexLabelVec lbls):
    tensor{std::move(t)}, labels{std::move(lbls)} {}

  LocalTensor<T>& raw_tensor() const {
    if(!tensor) throw py::value_error("Invalid labeled local tensor: missing tensor");
    return tensor->raw();
  }

  LabeledTensor<T> materialize() const {
    if(!tensor) throw py::value_error("Invalid labeled local tensor: missing tensor");
    return LabeledTensor<T>{tensor->raw(), labels};
  }
};

template<typename T>
static LocalTensor<T> make_local_tensor_from_sequence(py::sequence seq) {
  const auto n = py::len(seq);
  if(n == 0) return LocalTensor<T>{};

  bool all_tis = true;
  bool all_lbl = true;
  bool all_int = true;

  for(auto item: seq) {
    py::handle h = item;
    all_tis      = all_tis && py::isinstance<TiledIndexSpace>(h);
    all_lbl      = all_lbl && py::isinstance<TiledIndexLabel>(h);
    all_int      = all_int && py::isinstance<py::int_>(h);
  }

  if(all_tis) {
    TiledIndexSpaceVec spaces;
    spaces.reserve(static_cast<size_t>(n));
    for(auto item: seq) spaces.push_back(item.cast<TiledIndexSpace>());
    return make_local_tensor_from_tis_vec<T>(spaces);
  }

  if(all_lbl) {
    IndexLabelVec labels;
    labels.reserve(static_cast<size_t>(n));
    for(auto item: seq) labels.push_back(item.cast<TiledIndexLabel>());
    return make_local_tensor_from_label_vec<T>(labels);
  }

  if(all_int) {
    std::vector<size_t> sizes;
    sizes.reserve(static_cast<size_t>(n));
    for(auto item: seq) sizes.push_back(item.cast<size_t>());
    return make_local_tensor_from_size_vec<T>(sizes);
  }

  throw py::type_error("LocalTensor sequence constructor expects all TiledIndexSpace, all "
                       "TiledIndexLabel, or all integer sizes");
}

template<typename T>
static std::shared_ptr<PyLabeledLocalTensor<T>>
make_labeled_local_tensor_from_args(std::shared_ptr<PyLocalTensor<T>> self, py::args args) {
  if(!self) throw py::value_error("LocalTensor object is null");

  if(args.size() == 0) {
    auto lt = self->raw()();
    return std::make_shared<PyLabeledLocalTensor<T>>(self, lt.labels());
  }

  bool all_lbl = true;
  bool all_str = true;
  for(auto a: args) {
    all_lbl = all_lbl && py::isinstance<TiledIndexLabel>(a);
    all_str = all_str && py::isinstance<py::str>(a);
  }

  if(all_lbl) {
    IndexLabelVec labels;
    labels.reserve(args.size());
    for(auto a: args) labels.push_back(a.cast<TiledIndexLabel>());
    return std::make_shared<PyLabeledLocalTensor<T>>(self, std::move(labels));
  }

  if(all_str) {
    const auto& spaces = self->raw().tiled_index_spaces();
    if(args.size() != spaces.size()) {
      throw py::value_error("Number of string labels must match tensor rank");
    }

    IndexLabelVec labels;
    labels.reserve(args.size());
    for(size_t i = 0; i < args.size(); ++i) {
      labels.push_back(spaces[i].string_label(args[i].cast<std::string>()));
    }

    return std::make_shared<PyLabeledLocalTensor<T>>(self, std::move(labels));
  }

  throw py::type_error("LocalTensor.__call__ expects all arguments to be strings or "
                       "TiledIndexLabel objects");
}

template<typename T>
static LocalTensor<T>& raw_local_tensor(const std::shared_ptr<PyLocalTensor<T>>& t) {
  if(!t) throw py::value_error("LocalTensor must not be None");
  return t->raw();
}

// -----------------------------------------------------------------------------
// Tensor-expression layer
// Dedicated classes for handling expressions
// Necessary for ensuring compliance with TAMM's operator structure
// -----------------------------------------------------------------------------

using AnyPyLT = std::variant<std::shared_ptr<PLTd>, std::shared_ptr<PLTz>>;

struct PyTensorExpr {
  enum class Kind { Tensor, Scale, Mul };

  Kind                          kind = Kind::Tensor;
  AnyPyLT                       tensor{};
  PyScalar                      alpha{0.0};
  std::shared_ptr<PyTensorExpr> lhs;
  std::shared_ptr<PyTensorExpr> rhs;

  static PyTensorExpr from_tensor(const std::shared_ptr<PLTd>& t) {
    if(!t) throw py::value_error("LabeledTensorDouble must not be None");
    PyTensorExpr e;
    e.kind   = Kind::Tensor;
    e.tensor = t;
    return e;
  }

  static PyTensorExpr from_tensor(const std::shared_ptr<PLTz>& t) {
    if(!t) throw py::value_error("LabeledTensorComplexDouble must not be None");
    PyTensorExpr e;
    e.kind   = Kind::Tensor;
    e.tensor = t;
    return e;
  }

  static PyTensorExpr scaled(PyScalar a, const PyTensorExpr& e) {
    PyTensorExpr out;
    out.kind  = Kind::Scale;
    out.alpha = std::move(a);
    out.lhs   = std::make_shared<PyTensorExpr>(e);
    return out;
  }

  static PyTensorExpr mul(const PyTensorExpr& a, const PyTensorExpr& b) {
    PyTensorExpr out;
    out.kind = Kind::Mul;
    out.lhs  = std::make_shared<PyTensorExpr>(a);
    out.rhs  = std::make_shared<PyTensorExpr>(b);
    return out;
  }
};

struct NormalizedExpr {
  std::optional<PyScalar> alpha;
  AnyPyLT                 a;
  std::optional<AnyPyLT>  b;
};

static PyTensorExpr py_to_tensor_expr(py::handle h) {
  if(py::isinstance<PyTensorExpr>(h)) return h.cast<const PyTensorExpr&>();
  if(py::isinstance<PLTd>(h)) return PyTensorExpr::from_tensor(h.cast<std::shared_ptr<PLTd>>());
  if(py::isinstance<PLTz>(h)) return PyTensorExpr::from_tensor(h.cast<std::shared_ptr<PLTz>>());
  throw py::type_error("Expected TensorExpr, LabeledTensorDouble, or "
                       "LabeledTensorComplexDouble");
}

static PyScalar multiply_scalars(const PyScalar& x, const PyScalar& y) {
  if(std::holds_alternative<double>(x) && std::holds_alternative<double>(y)) {
    return std::get<double>(x) * std::get<double>(y);
  }

  const auto cx = std::holds_alternative<double>(x) ? std::complex<double>{std::get<double>(x), 0.0}
                                                    : std::get<std::complex<double>>(x);
  const auto cy = std::holds_alternative<double>(y) ? std::complex<double>{std::get<double>(y), 0.0}
                                                    : std::get<std::complex<double>>(y);

  return cx * cy;
}

static NormalizedExpr normalize_tensor_expr_impl(const PyTensorExpr& e) {
  using Kind = PyTensorExpr::Kind;

  if(e.kind == Kind::Tensor) return {std::nullopt, e.tensor, std::nullopt};

  if(e.kind == Kind::Scale) {
    if(!e.lhs) throw py::value_error("Malformed TensorExpr: missing scale child");

    auto inner = normalize_tensor_expr_impl(*e.lhs);
    if(inner.alpha.has_value()) inner.alpha = multiply_scalars(*inner.alpha, e.alpha);
    else inner.alpha = e.alpha;
    return inner;
  }

  if(e.kind == Kind::Mul) {
    if(!e.lhs || !e.rhs) throw py::value_error("Malformed TensorExpr: missing multiply child");

    auto l = normalize_tensor_expr_impl(*e.lhs);
    auto r = normalize_tensor_expr_impl(*e.rhs);

    if(l.b.has_value() || r.b.has_value()) {
      throw py::value_error("Unsupported expression. Only A, alpha*A, A*B, and alpha*A*B "
                            "are supported");
    }

    std::optional<PyScalar> alpha = l.alpha;
    if(r.alpha.has_value()) alpha = alpha ? multiply_scalars(*alpha, *r.alpha) : r.alpha;

    return {alpha, l.a, r.a};
  }

  throw py::value_error("Unknown TensorExpr kind");
}

static NormalizedExpr normalize_tensor_expr(const PyTensorExpr& e) {
  return normalize_tensor_expr_impl(e);
}

// -----------------------------------------------------------------------------
// LabeledTensor make_op bridge
// TAMM's Op handling is highly sensitive
// These methods ensure Op objects are handled correctly
// -----------------------------------------------------------------------------

struct PyOpExpr {
  OpList ops;
  OpList canonicalize() const { return ops; }
};

template<typename X>
inline constexpr bool is_ltd_type_v = std::is_same_v<std::decay_t<X>, LTd>;

template<typename X>
inline constexpr bool is_ltz_type_v = std::is_same_v<std::decay_t<X>, LTz>;

template<typename X>
inline constexpr bool is_ltd_or_ltz_type_v = is_ltd_type_v<X> || is_ltz_type_v<X>;

template<typename X>
inline constexpr bool is_expr_scalar_type_v =
  std::is_same_v<std::decay_t<X>, double> || std::is_same_v<std::decay_t<X>, std::complex<double>>;

template<typename LhsT, typename RhsT>
inline constexpr bool make_op_direct_lt_supported_v =
  (std::is_same_v<LhsT, double> ||
   std::is_same_v<LhsT, std::complex<double>>) &&is_ltd_or_ltz_type_v<RhsT>;

template<typename LhsT, typename AlphaT, typename RhsT>
inline constexpr bool
  make_op_scaled_lt_supported_v = (std::is_same_v<LhsT, double> &&
                                   std::is_same_v<std::decay_t<AlphaT>, double> &&
                                   is_ltd_type_v<RhsT>) ||
                                  (std::is_same_v<LhsT, std::complex<double>> &&
                                   is_expr_scalar_type_v<AlphaT> && is_ltd_or_ltz_type_v<RhsT>);

template<typename LhsT, typename AT, typename BT>
inline constexpr bool make_op_product_supported_v = (std::is_same_v<LhsT, double> &&
                                                     ((is_ltd_type_v<AT> && is_ltd_type_v<BT>) ||
                                                      (is_ltd_type_v<AT> && is_ltz_type_v<BT>) ||
                                                      (is_ltz_type_v<AT> && is_ltd_type_v<BT>) )) ||
                                                    (std::is_same_v<LhsT, std::complex<double>> &&
                                                     is_ltd_or_ltz_type_v<AT> &&
                                                     is_ltd_or_ltz_type_v<BT>);

template<typename LhsT, typename AlphaT, typename AT, typename BT>
inline constexpr bool make_op_scaled_product_supported_v =
  (std::is_same_v<LhsT, double> && std::is_same_v<std::decay_t<AlphaT>, double> &&
   make_op_product_supported_v<LhsT, AT, BT>) ||
  (std::is_same_v<LhsT, std::complex<double>> && is_expr_scalar_type_v<AlphaT> &&
   make_op_product_supported_v<LhsT, AT, BT>);

[[noreturn]] static void throw_make_op_unsupported(const std::string& detail) {
  throw py::value_error(
    "Expression not supported by TAMM LabeledTensor::make_op for this lhs/rhs type "
    "combination" +
    (detail.empty() ? std::string{} : ": " + detail));
}

template<typename LhsT, typename RhsT>
static PyOpExpr emit_rhs_op(LabeledTensor<LhsT>& lhs, const std::string& op, const RhsT& rhs) {
  auto rhs_copy = rhs;

  if(op == "=") return PyOpExpr{(lhs = rhs_copy).canonicalize()};
  if(op == "+=") return PyOpExpr{(lhs += rhs_copy).canonicalize()};
  if(op == "-=") return PyOpExpr{(lhs -= rhs_copy).canonicalize()};

  throw py::value_error("Invalid operator");
}

template<typename LhsT, typename A0, typename A1>
static PyOpExpr emit_tuple2_op(LabeledTensor<LhsT>& lhs, const std::string& op, const A0& a0,
                               const A1& a1) {
  auto tup = std::make_tuple(a0, a1);

  if(op == "=") return PyOpExpr{(lhs = tup).canonicalize()};
  if(op == "+=") return PyOpExpr{(lhs += tup).canonicalize()};
  if(op == "-=") return PyOpExpr{(lhs -= tup).canonicalize()};

  throw py::value_error("Invalid operator");
}

template<typename LhsT, typename A0, typename A1, typename A2>
static PyOpExpr emit_tuple3_op(LabeledTensor<LhsT>& lhs, const std::string& op, const A0& a0,
                               const A1& a1, const A2& a2) {
  auto tup = std::make_tuple(a0, a1, a2);

  if(op == "=") return PyOpExpr{(lhs = tup).canonicalize()};
  if(op == "+=") return PyOpExpr{(lhs += tup).canonicalize()};
  if(op == "-=") return PyOpExpr{(lhs -= tup).canonicalize()};

  throw py::value_error("Invalid operator");
}

template<typename LhsT>
static PyOpExpr emit_assign_like(PyLabeledTensor<LhsT>& lhs_proxy, const NormalizedExpr& n,
                                 const std::string& op) {
  auto lhs = lhs_proxy.materialize();

  if(!n.b.has_value()) {
    if(!n.alpha.has_value()) {
      return std::visit(
        [&](const auto& Ap) -> PyOpExpr {
          if(!Ap) throw py::value_error("Null labeled tensor in expression");

          auto A   = Ap->materialize();
          using AT = std::decay_t<decltype(A)>;

          if constexpr(make_op_direct_lt_supported_v<LhsT, AT>) { return emit_rhs_op(lhs, op, A); }
          else { throw_make_op_unsupported("direct labeled tensor"); }
        },
        n.a);
    }

    return std::visit(
      [&](const auto& Ap) -> PyOpExpr {
        if(!Ap) throw py::value_error("Null labeled tensor in expression");

        auto A   = Ap->materialize();
        using AT = std::decay_t<decltype(A)>;

        if(std::holds_alternative<double>(*n.alpha)) {
          const double alpha = std::get<double>(*n.alpha);
          if constexpr(make_op_scaled_lt_supported_v<LhsT, double, AT>) {
            return emit_tuple2_op(lhs, op, alpha, A);
          }
          else { throw_make_op_unsupported("scaled labeled tensor"); }
        }

        const auto alpha = std::get<std::complex<double>>(*n.alpha);
        if constexpr(make_op_scaled_lt_supported_v<LhsT, std::complex<double>, AT>) {
          return emit_tuple2_op(lhs, op, alpha, A);
        }
        else { throw_make_op_unsupported("complex-scaled labeled tensor"); }
      },
      n.a);
  }

  return std::visit(
    [&](const auto& Ap, const auto& Bp) -> PyOpExpr {
      if(!Ap || !Bp) throw py::value_error("Null labeled tensor in expression");

      auto A = Ap->materialize();
      auto B = Bp->materialize();

      using AT = std::decay_t<decltype(A)>;
      using BT = std::decay_t<decltype(B)>;

      if(!n.alpha.has_value()) {
        if constexpr(make_op_product_supported_v<LhsT, AT, BT>) {
          return emit_tuple2_op(lhs, op, A, B);
        }
        else { throw_make_op_unsupported("tensor product"); }
      }

      if(std::holds_alternative<double>(*n.alpha)) {
        const double alpha = std::get<double>(*n.alpha);
        if constexpr(make_op_scaled_product_supported_v<LhsT, double, AT, BT>) {
          return emit_tuple3_op(lhs, op, alpha, A, B);
        }
        else { throw_make_op_unsupported("scaled tensor product"); }
      }

      const auto alpha = std::get<std::complex<double>>(*n.alpha);
      if constexpr(make_op_scaled_product_supported_v<LhsT, std::complex<double>, AT, BT>) {
        return emit_tuple3_op(lhs, op, alpha, A, B);
      }
      else { throw_make_op_unsupported("complex-scaled tensor product"); }
    },
    n.a, *n.b);
}

template<typename T>
static PyOpExpr make_py_op_expr_from_expr(PyLabeledTensor<T>& lhs, const std::string& op,
                                          const PyTensorExpr& expr) {
  if(!(op == "=" || op == "+=" || op == "-=")) {
    throw py::value_error("Operator must be one of '=', '+=', '-='");
  }
  return emit_assign_like(lhs, normalize_tensor_expr(expr), op);
}

template<typename T>
static PyOpExpr make_py_op_expr_from_scalar(PyLabeledTensor<T>& lhs, const std::string& op,
                                            const T& value) {
  auto raw_lhs = lhs.materialize();
  return emit_rhs_op(raw_lhs, op, value);
}

// -----------------------------------------------------------------------------
// DAG helpers
// DAG involves difficult-to-bind operator expressions, so we use dedicated Python classes
// -----------------------------------------------------------------------------

using AnyPyTensor = std::variant<std::shared_ptr<PTd>, std::shared_ptr<PTz>>;

static const void* tensor_key(const std::shared_ptr<PTd>& t) {
  return static_cast<const void*>(t.get());
}

static const void* tensor_key(const std::shared_ptr<PTz>& t) {
  return static_cast<const void*>(t.get());
}

struct PyTensorCollector {
  std::set<const void*>    keys;
  std::vector<AnyPyTensor> tensors;

  void add(const std::shared_ptr<PTd>& t) {
    if(!t) return;
    if(keys.insert(tensor_key(t)).second) tensors.emplace_back(t);
  }

  void add(const std::shared_ptr<PTz>& t) {
    if(!t) return;
    if(keys.insert(tensor_key(t)).second) tensors.emplace_back(t);
  }

  bool contains(const AnyPyTensor& t) const {
    return std::visit([&](const auto& p) { return keys.find(tensor_key(p)) != keys.end(); }, t);
  }
};

static void collect_expr_tensors(const PyTensorExpr& e, PyTensorCollector& c);

static void collect_labeled_tensor(const AnyPyLT& lt, PyTensorCollector& c) {
  std::visit(
    [&](const auto& p) {
      if(p && p->tensor) c.add(p->tensor);
    },
    lt);
}

static void collect_expr_tensors(const PyTensorExpr& e, PyTensorCollector& c) {
  if(e.kind == PyTensorExpr::Kind::Tensor) {
    collect_labeled_tensor(e.tensor, c);
    return;
  }

  if(e.lhs) collect_expr_tensors(*e.lhs, c);
  if(e.rhs) collect_expr_tensors(*e.rhs, c);
}

static void collect_pyobject_tensors(py::handle obj, PyTensorCollector& c) {
  if(obj.is_none()) return;

  if(py::isinstance<PTd>(obj)) {
    c.add(obj.cast<std::shared_ptr<PTd>>());
    return;
  }

  if(py::isinstance<PTz>(obj)) {
    c.add(obj.cast<std::shared_ptr<PTz>>());
    return;
  }

  if(py::isinstance<PLTd>(obj)) {
    auto lt = obj.cast<std::shared_ptr<PLTd>>();
    if(lt && lt->tensor) c.add(lt->tensor);
    return;
  }

  if(py::isinstance<PLTz>(obj)) {
    auto lt = obj.cast<std::shared_ptr<PLTz>>();
    if(lt && lt->tensor) c.add(lt->tensor);
    return;
  }

  if(py::isinstance<PyTensorExpr>(obj)) {
    collect_expr_tensors(obj.cast<const PyTensorExpr&>(), c);
    return;
  }

  if(py::isinstance<py::tuple>(obj) || py::isinstance<py::list>(obj)) {
    for(auto item: py::reinterpret_borrow<py::iterable>(obj)) collect_pyobject_tensors(item, c);
  }
}

static void allocate_collected_tensors(PyTensorCollector& c, ExecutionContext& ec) {
  for(auto& t: c.tensors) {
    std::visit(
      [&](auto& p) {
        if(p && !p->raw().is_allocated()) p->raw().allocate(&ec);
      },
      t);
  }
}

static void deallocate_tensor_variant(AnyPyTensor& t) {
  std::visit(
    [](auto& p) {
      if(p && p->raw().is_allocated()) p->raw().deallocate();
    },
    t);
}

struct PyDAG {
  py::function       func;
  py::tuple          args;
  mutable py::object materialized;
};

template<typename T>
static PyOpExpr make_py_op_expr_from_spec(PyLabeledTensor<T>& lhs, py::args args);

static OpList py_specs_to_oplist(py::iterable specs) {
  OpList all_ops;

  for(auto item: specs) {
    py::tuple spec = py::cast<py::tuple>(item);
    if(py::len(spec) < 3) {
      throw py::value_error("Each DAG op spec must be a tuple like (lhs, op, ...rhs...)");
    }

    py::object lhs_obj = spec[0];

    py::tuple tail(py::len(spec) - 1);
    for(py::ssize_t i = 1; i < py::len(spec); ++i) tail[i - 1] = spec[i];

    py::args rhs_args = py::reinterpret_borrow<py::args>(tail);

    PyOpExpr expr;
    if(py::isinstance<PLTd>(lhs_obj)) {
      auto lhs = lhs_obj.cast<std::shared_ptr<PLTd>>();
      if(!lhs) throw py::value_error("DAG lhs LabeledTensorDouble is None");
      expr = make_py_op_expr_from_spec<double>(*lhs, rhs_args);
    }
    else if(py::isinstance<PLTz>(lhs_obj)) {
      auto lhs = lhs_obj.cast<std::shared_ptr<PLTz>>();
      if(!lhs) throw py::value_error("DAG lhs LabeledTensorComplexDouble is None");
      expr = make_py_op_expr_from_spec<std::complex<double>>(*lhs, rhs_args);
    }
    else {
      throw py::type_error("DAG lhs must be LabeledTensorDouble or "
                           "LabeledTensorComplexDouble");
    }

    for(const auto& op: expr.ops) all_ops.push_back(op);
  }

  return all_ops;
}

static OpList pydag_to_oplist(PyDAG& dag) {
  py::gil_scoped_acquire gil;

  dag.materialized = dag.func(*dag.args);

  py::object specs_obj = dag.materialized;
  if(py::isinstance<py::tuple>(dag.materialized)) {
    auto tup = dag.materialized.cast<py::tuple>();
    if(py::len(tup) >= 1) specs_obj = tup[0];
  }

  return py_specs_to_oplist(specs_obj.cast<py::iterable>());
}

// -----------------------------------------------------------------------------
// Scheduler op-spec parser
// Generates TAMM operations based on Python call
// TAMM syntax (lhs = alpha * a * b)
// Python equivalent (lhs, "=", alpha, a * b)
// -----------------------------------------------------------------------------

template<typename T>
static PyOpExpr make_py_op_expr_from_spec(PyLabeledTensor<T>& lhs, py::args args) {
  if(args.size() < 1) {
    throw py::value_error("Scheduler tensor operation requires at least an operator string");
  }

  if(!py::isinstance<py::str>(args[0])) {
    throw py::type_error("First argument after lhs must be an operator string");
  }

  const std::string op = args[0].cast<std::string>();
  if(!(op == "=" || op == "+=" || op == "-=")) {
    throw py::value_error("Supported operators are '=', '+=', '-='");
  }

  auto is_lt_any = [](py::handle h) { return py::isinstance<PLTd>(h) || py::isinstance<PLTz>(h); };

  if(args.size() == 2 && py_is_scalar_compatible<T>(args[1])) {
    return make_py_op_expr_from_scalar(lhs, op, py_numeric_scalar_cast<T>(args[1]));
  }

  if(args.size() == 2 && (py::isinstance<PyTensorExpr>(args[1]) || is_lt_any(args[1]))) {
    return make_py_op_expr_from_expr(lhs, op, py_to_tensor_expr(args[1]));
  }

  if(args.size() == 3 && is_lt_any(args[1]) && is_lt_any(args[2])) {
    auto expr = PyTensorExpr::mul(py_to_tensor_expr(args[1]), py_to_tensor_expr(args[2]));
    return make_py_op_expr_from_expr(lhs, op, expr);
  }

  if(args.size() == 3 && py_is_numeric_scalar(args[1]) && is_lt_any(args[2])) {
    auto expr = PyTensorExpr::scaled(py_to_expr_scalar(args[1]), py_to_tensor_expr(args[2]));
    return make_py_op_expr_from_expr(lhs, op, expr);
  }

  if(args.size() == 4 && py_is_numeric_scalar(args[1]) && is_lt_any(args[2]) &&
     is_lt_any(args[3])) {
    auto expr = PyTensorExpr::scaled(
      py_to_expr_scalar(args[1]),
      PyTensorExpr::mul(py_to_tensor_expr(args[2]), py_to_tensor_expr(args[3])));
    return make_py_op_expr_from_expr(lhs, op, expr);
  }

  throw py::value_error("Unsupported scheduler tensor-op form. Supported forms include "
                        "(lhs, '=', value), (lhs, '=', A), (lhs, '=', A*B), "
                        "(lhs, '=', alpha*A), and (lhs, '=', alpha*A*B), with += and -= variants.");
}

// -----------------------------------------------------------------------------
// Type names
// -----------------------------------------------------------------------------

template<typename T>
struct PyTypeInfo;

template<>
struct PyTypeInfo<double> {
  static constexpr const char* suffix    = "Double";
  static constexpr const char* fill_name = "fill_sparse_tensor_double";
};

template<>
struct PyTypeInfo<std::complex<double>> {
  static constexpr const char* suffix    = "ComplexDouble";
  static constexpr const char* fill_name = "fill_sparse_tensor_complex_double";
};

template<typename T>
static std::string tensor_class_name() {
  return std::string("Tensor") + PyTypeInfo<T>::suffix;
}

template<typename T>
static std::string labeled_tensor_class_name() {
  return std::string("LabeledTensor") + PyTypeInfo<T>::suffix;
}

// -----------------------------------------------------------------------------
// new_ops helpers
// -----------------------------------------------------------------------------

static std::unique_ptr<NewOpBase> clone_new_op_unique_checked(const NewOpBase& op,
                                                              const char*      where) {
  auto up = op.clone();
  if(!up) throw py::type_error(std::string(where) + ": op.clone() returned nullptr");
  return up;
}

static std::shared_ptr<NewOpBase> unique_op_to_shared_base(std::unique_ptr<NewOpBase>&& up) {
  if(!up) throw py::type_error("unique_op_to_shared_base received nullptr");
  return std::shared_ptr<NewOpBase>(std::move(up));
}

static py::object clone_new_op_py(const NewOpBase& op) {
  if(const auto* p = dynamic_cast<const NewLTOp*>(&op))
    return py::cast(std::make_shared<NewLTOp>(*p));
  if(const auto* p = dynamic_cast<const NewMultOp*>(&op))
    return py::cast(std::make_shared<NewMultOp>(*p));
  if(const auto* p = dynamic_cast<const NewAddOp*>(&op))
    return py::cast(std::make_shared<NewAddOp>(*p));

  return py::cast(unique_op_to_shared_base(clone_new_op_unique_checked(op, "clone_new_op_py")));
}

static std::unique_ptr<NewOpBase> scale_new_op_unique(const NewOpBase&    op,
                                                      const tamm::Scalar& alpha) {
  if(const auto* p = dynamic_cast<const NewLTOp*>(&op)) return std::make_unique<NewLTOp>(*p, alpha);
  if(const auto* p = dynamic_cast<const NewMultOp*>(&op))
    return std::make_unique<NewMultOp>(*p, alpha);
  if(const auto* p = dynamic_cast<const NewAddOp*>(&op))
    return std::make_unique<NewAddOp>(*p, alpha);

  auto out = clone_new_op_unique_checked(op, "scale_new_op_unique fallback");
  out->set_coeff(out->coeff() * alpha);
  return out;
}

static py::object scale_new_op_py(const NewOpBase& op, const tamm::Scalar& alpha) {
  if(const auto* p = dynamic_cast<const NewLTOp*>(&op))
    return py::cast(std::make_shared<NewLTOp>(*p, alpha));
  if(const auto* p = dynamic_cast<const NewMultOp*>(&op))
    return py::cast(std::make_shared<NewMultOp>(*p, alpha));
  if(const auto* p = dynamic_cast<const NewAddOp*>(&op))
    return py::cast(std::make_shared<NewAddOp>(*p, alpha));

  return py::cast(unique_op_to_shared_base(scale_new_op_unique(op, alpha)));
}

static std::shared_ptr<NewMultOp> make_mult_expr_typed(const NewOpBase& lhs, const NewOpBase& rhs) {
  return std::make_shared<NewMultOp>(clone_new_op_unique_checked(lhs, "make_mult_expr_typed lhs"),
                                     clone_new_op_unique_checked(rhs, "make_mult_expr_typed rhs"));
}

static std::shared_ptr<NewAddOp> make_add_expr_typed(const NewOpBase& lhs, const NewOpBase& rhs) {
  return std::make_shared<NewAddOp>(clone_new_op_unique_checked(lhs, "make_add_expr_typed lhs"),
                                    clone_new_op_unique_checked(rhs, "make_add_expr_typed rhs"));
}

static std::shared_ptr<NewAddOp> make_sub_expr_typed(const NewOpBase& lhs, const NewOpBase& rhs) {
  return std::make_shared<NewAddOp>(clone_new_op_unique_checked(lhs, "make_sub_expr_typed lhs"),
                                    scale_new_op_unique(rhs, tamm::Scalar{-1.0}));
}

static NewLTOp make_ltop_value_from_labeled_double(const std::shared_ptr<PLTd>& lt) {
  if(!lt) throw py::value_error("LabeledTensorDouble must not be None");
  return NewLTOp{lt->materialize()};
}

static std::shared_ptr<NewLTOp>
make_ltop_expr_from_labeled_double_typed(const std::shared_ptr<PLTd>& lt) {
  if(!lt) throw py::value_error("LabeledTensorDouble must not be None");
  return std::make_shared<NewLTOp>(lt->materialize());
}

static std::shared_ptr<NewOpBase>
make_ltop_expr_from_labeled_double(const std::shared_ptr<PLTd>& lt) {
  return std::static_pointer_cast<NewOpBase>(make_ltop_expr_from_labeled_double_typed(lt));
}

static std::shared_ptr<PLTd>
update_labeled_double_with_new_op(const std::shared_ptr<PLTd>&      lhs,
                                  const std::shared_ptr<NewOpBase>& rhs) {
  if(!lhs) throw py::value_error("lhs LabeledTensorDouble must not be None");
  if(!rhs) throw py::value_error("rhs ExprOp must not be None");

  auto lhs_lt = lhs->materialize();
  lhs_lt.update(*rhs);

  return lhs;
}

// -----------------------------------------------------------------------------
// Tensor raw-access helpers
// -----------------------------------------------------------------------------

template<typename T>
static Tensor<T>& raw_tensor(const std::shared_ptr<PyTensor<T>>& t) {
  if(!t) throw py::value_error("Tensor must not be None");
  return t->raw();
}

template<typename T>
static LabeledTensor<T> raw_labeled(const std::shared_ptr<PyLabeledTensor<T>>& lt) {
  if(!lt) throw py::value_error("LabeledTensor must not be None");
  return lt->materialize();
}

// -----------------------------------------------------------------------------
// NumPy conversions
// -----------------------------------------------------------------------------

template<typename T>
using PyCArray = py::array_t<T, py::array::c_style | py::array::forcecast>;

static std::vector<size_t> numpy_shape(py::array arr, const char* where) {
  if(arr.ndim() == 0) {
    throw py::value_error(std::string(where) + ": scalar NumPy arrays are not supported");
  }

  std::vector<size_t> shape;
  shape.reserve(static_cast<size_t>(arr.ndim()));

  for(py::ssize_t i = 0; i < arr.ndim(); ++i) {
    if(arr.shape(i) < 0) throw py::value_error(std::string(where) + ": invalid shape");
    shape.push_back(static_cast<size_t>(arr.shape(i)));
  }

  return shape;
}

static TiledIndexSpaceVec make_tiled_spaces_from_numpy_shape(const std::vector<size_t>& shape,
                                                             size_t                     tilesize) {
  if(tilesize == 0) throw py::value_error("tilesize must be greater than zero");

  TiledIndexSpaceVec spaces;
  spaces.reserve(shape.size());

  for(size_t n: shape) {
    if(n > static_cast<size_t>(std::numeric_limits<Index>::max())) {
      throw py::value_error("NumPy dimension is too large for TAMM Index");
    }

    const auto is   = IndexSpace{range(static_cast<Index>(n))};
    const auto tile = static_cast<Tile>(std::min(std::max<size_t>(n, 1), tilesize));
    spaces.emplace_back(is, tile);
  }

  return spaces;
}

static TiledIndexSpaceVec
numpy_spaces_or_default(py::object spaces_obj, const std::vector<size_t>& shape, size_t tilesize) {
  if(spaces_obj.is_none()) return make_tiled_spaces_from_numpy_shape(shape, tilesize);

  auto spaces = spaces_obj.cast<TiledIndexSpaceVec>();

  if(spaces.size() != shape.size()) {
    throw py::value_error("spaces length must match NumPy array rank");
  }

  for(size_t i = 0; i < shape.size(); ++i) {
    const size_t n = static_cast<size_t>(spaces[i].max_num_indices());
    if(n != shape[i]) {
      throw py::value_error("spaces[" + std::to_string(i) + "] has size " + std::to_string(n) +
                            ", but NumPy dimension is " + std::to_string(shape[i]));
    }
  }

  return spaces;
}

template<typename T>
static std::vector<size_t> tamm_dense_shape(const Tensor<T>& tensor) {
  const auto& spaces = tensor.tiled_index_spaces();

  std::vector<size_t> shape;
  shape.reserve(spaces.size());

  for(const auto& tis: spaces) { shape.push_back(static_cast<size_t>(tis.max_num_indices())); }

  return shape;
}

template<typename T>
static void copy_numpy_to_tamm_tensor(Tensor<T>& tensor, PyCArray<T> array) {
  if(!tensor.is_allocated()) {
    throw py::value_error("copy_from_numpy target tensor must be allocated");
  }

  const auto arr_shape = numpy_shape(array, "copy_from_numpy");
  const auto ten_shape = tamm_dense_shape(tensor);

  if(arr_shape != ten_shape) {
    throw py::value_error("copy_from_numpy shape mismatch between NumPy array and TAMM tensor");
  }

  const size_t nmodes = arr_shape.size();
  const T*     dense  = array.data();

  std::vector<size_t> dense_strides(nmodes, 1);
  for(size_t i = nmodes; i-- > 0;) {
    dense_strides[i] = (i + 1 < nmodes) ? dense_strides[i + 1] * arr_shape[i + 1] : 1;
  }

  auto lt = tensor();

  {
    py::gil_scoped_release release;

    tamm::update_tensor<T>(lt, [&](const IndexVector& blockid, auto&& buf) {
      auto block_dims   = tensor.block_dims(blockid);
      auto block_offset = tensor.block_offsets(blockid);

      if(block_dims.size() != nmodes || block_offset.size() != nmodes) {
        throw std::runtime_error("copy_from_numpy: rank mismatch in TAMM block metadata");
      }

      for(size_t c = 0; c < static_cast<size_t>(buf.size()); ++c) {
        size_t tmp       = c;
        size_t src_index = 0;

        for(size_t d = nmodes; d-- > 0;) {
          const size_t bd = static_cast<size_t>(block_dims[d]);
          const size_t lc = bd == 0 ? 0 : tmp % bd;
          if(bd != 0) tmp /= bd;

          const size_t gc = static_cast<size_t>(block_offset[d]) + lc;
          src_index += gc * dense_strides[d];
        }

        buf[c] = dense[src_index];
      }
    });
  }
}

template<typename T>
static std::shared_ptr<PyTensor<T>> numpy_to_tamm_tensor(py::array array, ExecutionContext& ec,
                                                         size_t     tilesize,
                                                         py::object spaces_obj = py::none()) {
  auto typed = PyCArray<T>::ensure(array);
  if(!typed) throw py::type_error("Could not convert input to a C-contiguous NumPy array");

  const auto shape  = numpy_shape(typed, "from_numpy");
  auto       spaces = numpy_spaces_or_default(spaces_obj, shape, tilesize);

  auto tensor = std::make_shared<PyTensor<T>>(Tensor<T>{spaces});
  tensor->raw().allocate(&ec);

  copy_numpy_to_tamm_tensor<T>(tensor->raw(), typed);
  return tensor;
}

static bool numpy_dtype_is_complex(py::array array) {
  py::module_ np = py::module_::import("numpy");
  return py::cast<bool>(np.attr("issubdtype")(array.attr("dtype"), np.attr("complexfloating")));
}

static py::object numpy_to_tamm_tensor_auto(py::array array, ExecutionContext& ec, size_t tilesize,
                                            py::object spaces_obj = py::none()) {
  if(numpy_dtype_is_complex(array)) {
    return py::cast(numpy_to_tamm_tensor<std::complex<double>>(array, ec, tilesize, spaces_obj));
  }

  return py::cast(numpy_to_tamm_tensor<double>(array, ec, tilesize, spaces_obj));
}

template<typename T>
static py::array_t<T> tamm_tensor_to_numpy_array(const tamm::Tensor<T>& tensor) {
  if(!tensor.is_allocated()) {
    throw py::value_error("Cannot convert an unallocated TAMM tensor to a NumPy array");
  }

  const size_t nmodes = static_cast<size_t>(tensor.num_modes());
  const auto&  tiss   = tensor.tiled_index_spaces();

  std::vector<size_t>      dims;
  std::vector<py::ssize_t> shape;
  dims.reserve(nmodes);
  shape.reserve(nmodes);

  for(size_t i = 0; i < nmodes; ++i) {
    push_checked_dim(dims, shape, static_cast<size_t>(tiss[i].max_num_indices()),
                     "tamm_tensor_to_numpy_array");
  }

  const size_t total = checked_product_or_one(dims, "tamm_tensor_to_numpy_array");

  py::array_t<T> arr(shape);
  T*             data = arr.mutable_data();

  if(total > 0) std::fill_n(data, total, T{});

  std::vector<size_t> strides(nmodes, 1);
  for(size_t i = nmodes; i-- > 0;) {
    strides[i] = (i + 1 < nmodes) ? strides[i + 1] * dims[i + 1] : 1;
  }

  {
    py::gil_scoped_release release;

    for(const auto& blockid: tensor.loop_nest()) {
      const auto     bsz = tensor.block_size(blockid);
      std::vector<T> buf(static_cast<size_t>(bsz));

      tensor.get(blockid, buf);

      auto block_dims   = tensor.block_dims(blockid);
      auto block_offset = tensor.block_offsets(blockid);

      if(block_dims.size() != nmodes || block_offset.size() != nmodes) {
        throw std::runtime_error("tamm_tensor_to_numpy_array: rank mismatch in block metadata");
      }

      if(nmodes == 0) {
        if(!buf.empty()) data[0] = buf[0];
        continue;
      }

      for(size_t c = 0; c < buf.size(); ++c) {
        size_t tmp       = c;
        size_t dst_index = 0;

        for(size_t d = nmodes; d-- > 0;) {
          const size_t bd = static_cast<size_t>(block_dims[d]);
          const size_t lc = bd == 0 ? 0 : tmp % bd;
          if(bd != 0) tmp /= bd;

          const size_t gc = static_cast<size_t>(block_offset[d]) + lc;
          if(gc >= dims[d]) {
            throw std::runtime_error("tamm_tensor_to_numpy_array: block exceeds dense shape");
          }

          dst_index += gc * strides[d];
        }

        data[dst_index] = buf[c];
      }
    }
  }

  return arr;
}

template<typename T>
static py::array_t<T> tamm_local_tensor_to_numpy_array(tamm::LocalTensor<T>& tensor) {
  if(!tensor.is_allocated()) {
    throw py::value_error("Cannot convert an unallocated TAMM LocalTensor to a NumPy array");
  }

  const auto dim_sizes = tensor.dim_sizes();

  std::vector<size_t>      dims;
  std::vector<py::ssize_t> shape;
  dims.reserve(dim_sizes.size());
  shape.reserve(dim_sizes.size());

  for(size_t n: dim_sizes) { push_checked_dim(dims, shape, n, "tamm_local_tensor_to_numpy_array"); }

  const size_t total = checked_product_or_one(dims, "tamm_local_tensor_to_numpy_array");

  py::array_t<T> arr(shape);
  T*             dst = arr.mutable_data();
  T*             src = tensor.access_local_buf();

  if(total > 0) {
    if(src == nullptr) throw py::value_error("LocalTensor local buffer is null");

    py::gil_scoped_release release;
    std::copy_n(src, total, dst);
  }

  return arr;
}

static void bind_numpy_conversions(py::module_& m) {
  m.def(
    "tiled_spaces",
    [](py::sequence shape, size_t tilesize) {
      std::vector<size_t> dims;
      dims.reserve(py::len(shape));

      for(auto item: shape) dims.push_back(item.cast<size_t>());

      return make_tiled_spaces_from_numpy_shape(dims, tilesize);
    },
    py::arg("shape"), py::arg("tilesize") = 32);

  m.def(
    "from_numpy",
    [](py::array array, ExecutionContext& ec, size_t tilesize, py::object spaces) {
      return numpy_to_tamm_tensor_auto(array, ec, tilesize, spaces);
    },
    py::arg("array"), py::arg("ec"), py::arg("tilesize") = 32, py::arg("spaces") = py::none());

  m.def(
    "from_numpy",
    [](py::array array, size_t tilesize, ExecutionContext& ec, py::object spaces) {
      return numpy_to_tamm_tensor_auto(array, ec, tilesize, spaces);
    },
    py::arg("array"), py::arg("tilesize"), py::arg("ec"), py::arg("spaces") = py::none());

  m.def(
    "asarray",
    [](py::array array, ExecutionContext& ec, size_t tilesize, py::object spaces) {
      return numpy_to_tamm_tensor_auto(array, ec, tilesize, spaces);
    },
    py::arg("array"), py::arg("ec"), py::arg("tilesize") = 32, py::arg("spaces") = py::none());

  m.def(
    "from_numpy_double",
    [](py::array array, ExecutionContext& ec, size_t tilesize, py::object spaces) {
      return numpy_to_tamm_tensor<double>(array, ec, tilesize, spaces);
    },
    py::arg("array"), py::arg("ec"), py::arg("tilesize") = 32, py::arg("spaces") = py::none());

  m.def(
    "from_numpy_complex_double",
    [](py::array array, ExecutionContext& ec, size_t tilesize, py::object spaces) {
      return numpy_to_tamm_tensor<std::complex<double>>(array, ec, tilesize, spaces);
    },
    py::arg("array"), py::arg("ec"), py::arg("tilesize") = 32, py::arg("spaces") = py::none());

  m.def(
    "copy_from_numpy",
    [](std::shared_ptr<PTd> tensor, py::array array) {
      if(!tensor) throw py::value_error("copy_from_numpy: tensor must not be None");

      auto typed = PyCArray<double>::ensure(array);
      if(!typed) throw py::type_error("Could not convert input to float64 NumPy array");

      copy_numpy_to_tamm_tensor<double>(tensor->raw(), typed);
      return tensor;
    },
    py::arg("tensor"), py::arg("array"));

  m.def(
    "copy_from_numpy",
    [](std::shared_ptr<PTz> tensor, py::array array) {
      if(!tensor) throw py::value_error("copy_from_numpy: tensor must not be None");

      auto typed = PyCArray<std::complex<double>>::ensure(array);
      if(!typed) throw py::type_error("Could not convert input to complex128 NumPy array");

      copy_numpy_to_tamm_tensor<std::complex<double>>(tensor->raw(), typed);
      return tensor;
    },
    py::arg("tensor"), py::arg("array"));
}

// -----------------------------------------------------------------------------
// Local tensor bindings
// -----------------------------------------------------------------------------

template<typename T>
void bind_local_tensor_family(py::module_& m, py::class_<Scheduler>& scheduler_cls) {
  using PyLocalTensorT  = PyLocalTensor<T>;
  using PyLabeledLocalT = PyLabeledLocalTensor<T>;

  const std::string tensor_name = std::string("LocalTensor") + PyTypeInfo<T>::suffix;
  const std::string lt_name     = std::string("LabeledLocalTensor") + PyTypeInfo<T>::suffix;

  py::class_<PyLabeledLocalT, std::shared_ptr<PyLabeledLocalT>>(m, lt_name.c_str())
    .def("labels", [](const PyLabeledLocalT& lt) { return lt.labels; })
    .def("tensor", [](const PyLabeledLocalT& lt) { return lt.tensor; })
    .def(
      "to_numpy",
      [](PyLabeledLocalT& lt, py::object dtype) {
        return numpy_array_protocol_result(tamm_local_tensor_to_numpy_array<T>(lt.raw_tensor()),
                                           dtype);
      },
      py::arg("dtype") = py::none())
    .def(
      "__array__",
      [](PyLabeledLocalT& lt, py::object dtype, py::object copy) {
        return numpy_array_protocol_result(tamm_local_tensor_to_numpy_array<T>(lt.raw_tensor()),
                                           dtype, copy);
      },
      py::arg("dtype") = py::none(), py::arg("copy") = py::none());

  py::class_<PyLocalTensorT, std::shared_ptr<PyLocalTensorT>>(m, tensor_name.c_str())
    .def(py::init([]() { return std::make_shared<PyLocalTensorT>(); }))
    .def(py::init([](py::sequence seq) {
      return std::make_shared<PyLocalTensorT>(make_local_tensor_from_sequence<T>(seq));
    }))
    .def("__call__",
         [](PyLocalTensorT& self, py::args args) {
           return make_labeled_local_tensor_from_args<T>(self.shared_from_this(), args);
         })
    .def(
      "allocate_self", [](PyLocalTensorT& t, ExecutionContext& ec) { t.allocate(ec); },
      py::arg("ec"))
    .def("deallocate", [](PyLocalTensorT& t) { t.deallocate(); })
    .def("is_allocated", &PyLocalTensorT::is_allocated)
    .def("num_modes", [](PyLocalTensorT& t) { return t.raw().num_modes(); })
    .def("tiled_index_spaces", [](PyLocalTensorT& t) { return t.raw().tiled_index_spaces(); })
    .def("dim_sizes", [](PyLocalTensorT& t) { return t.raw().dim_sizes(); })
    .def(
      "init", [](PyLocalTensorT& t, T value) { t.raw().init(value); }, py::arg("value"))
    .def("get",
         [](PyLocalTensorT& t, py::args args) {
           std::vector<size_t> indices;

           if(args.size() == 1 &&
              (py::isinstance<py::list>(args[0]) || py::isinstance<py::tuple>(args[0]))) {
             for(auto item: py::reinterpret_borrow<py::iterable>(args[0])) {
               indices.push_back(item.cast<size_t>());
             }
           }
           else {
             for(auto item: args) indices.push_back(item.cast<size_t>());
           }

           const LocalTensor<T>& cref = t.raw();
           return cref.get(indices);
         })
    .def(
      "set",
      [](PyLocalTensorT& t, py::sequence seq, T value) {
        std::vector<size_t> indices;
        for(auto item: seq) indices.push_back(item.cast<size_t>());
        t.raw().set(indices, value);
      },
      py::arg("indices"), py::arg("value"))
    .def(
      "resize",
      [](PyLocalTensorT& t, py::sequence seq) {
        std::vector<size_t> sizes;
        for(auto item: seq) sizes.push_back(item.cast<size_t>());
        t.raw().resize(sizes);
      },
      py::arg("sizes"))
    .def(
      "block",
      [](PyLocalTensorT& t, py::sequence start_offsets, py::sequence span_sizes) {
        std::vector<size_t> offsets;
        std::vector<size_t> spans;
        for(auto item: start_offsets) offsets.push_back(item.cast<size_t>());
        for(auto item: span_sizes) spans.push_back(item.cast<size_t>());
        return std::make_shared<PyLocalTensorT>(t.raw().block(offsets, spans));
      },
      py::arg("start_offsets"), py::arg("span_sizes"))
    .def(
      "block",
      [](PyLocalTensorT& t, size_t x_offset, size_t y_offset, size_t x_span, size_t y_span) {
        return std::make_shared<PyLocalTensorT>(t.raw().block(x_offset, y_offset, x_span, y_span));
      },
      py::arg("x_offset"), py::arg("y_offset"), py::arg("x_span"), py::arg("y_span"))
    .def(
      "from_distributed_tensor",
      [](PyLocalTensorT& local, std::shared_ptr<PyTensor<T>> dist) {
        if(!dist) throw py::value_error("Distributed tensor must not be None");
        local.raw().from_distributed_tensor(dist->raw());
      },
      py::arg("dist_tensor"))
    .def(
      "to_distributed_tensor",
      [](PyLocalTensorT& local, std::shared_ptr<PyTensor<T>> dist) {
        if(!dist) throw py::value_error("Distributed tensor must not be None");
        local.raw().to_distributed_tensor(dist->raw());
      },
      py::arg("dist_tensor"))
    .def("base_ptr",
         [](PyLocalTensorT& t) { return reinterpret_cast<std::uintptr_t>(t.raw().base_ptr()); })
    .def(
      "access_local_buf",
      [](PyLocalTensorT& t) -> py::object {
        auto* ptr = t.raw().access_local_buf();
        if(ptr == nullptr) return py::none();

        return py::array_t<T>({static_cast<py::ssize_t>(t.raw().local_buf_size())},
                              {static_cast<py::ssize_t>(sizeof(T))}, ptr, py::cast(&t));
      },
      py::return_value_policy::reference_internal)
    .def(
      "to_numpy",
      [](PyLocalTensorT& t, py::object dtype) {
        return numpy_array_protocol_result(tamm_local_tensor_to_numpy_array<T>(t.raw()), dtype);
      },
      py::arg("dtype") = py::none())
    .def(
      "__array__",
      [](PyLocalTensorT& t, py::object dtype, py::object copy) {
        return numpy_array_protocol_result(tamm_local_tensor_to_numpy_array<T>(t.raw()), dtype,
                                           copy);
      },
      py::arg("dtype") = py::none(), py::arg("copy") = py::none());

  m.def(
    "to_numpy",
    [](std::shared_ptr<PyLocalTensorT> t, py::object dtype) {
      if(!t) throw py::value_error("to_numpy: LocalTensor must not be None");
      return numpy_array_protocol_result(tamm_local_tensor_to_numpy_array<T>(t->raw()), dtype);
    },
    py::arg("tensor"), py::arg("dtype") = py::none());

  m.def(
    "to_numpy",
    [](std::shared_ptr<PyLabeledLocalT> t, py::object dtype) {
      if(!t) throw py::value_error("to_numpy: LabeledLocalTensor must not be None");
      return numpy_array_protocol_result(tamm_local_tensor_to_numpy_array<T>(t->raw_tensor()),
                                         dtype);
    },
    py::arg("tensor"), py::arg("dtype") = py::none());

  m.def(
    "print_tensor",
    [](std::shared_ptr<PyLocalTensorT> t, std::string filename) {
      tamm::print_tensor<T>(raw_local_tensor(t), filename);
    },
    py::arg("tensor"), py::arg("filename") = "");

  scheduler_cls.def(
    "__call__",
    [](Scheduler& s, std::shared_ptr<PyLabeledLocalT> lhs, py::args args,
       py::kwargs kwargs) -> Scheduler& {
      if(!lhs) throw py::value_error("Local tensor lhs must not be None");

      std::string opstr;
      ExecutionHW exhw = ExecutionHW::DEFAULT;

      if(kwargs) {
        if(kwargs.contains("opstr")) opstr = kwargs["opstr"].cast<std::string>();
        if(kwargs.contains("exhw")) exhw = kwargs["exhw"].cast<ExecutionHW>();
      }

      if(args.size() < 2) {
        throw py::value_error("Scheduler local tensor operation expects operator and rhs");
      }

      if(!py::isinstance<py::str>(args[0])) {
        throw py::type_error("First argument after lhs must be an operator string");
      }

      const std::string op = args[0].cast<std::string>();
      if(!(op == "=" || op == "+=" || op == "-=")) {
        throw py::value_error("Supported operators are '=', '+=', '-='");
      }

      auto lhs_lt = lhs->materialize();

      if(args.size() == 2 && py_is_scalar_compatible<T>(args[1])) {
        PyOpExpr expr = emit_rhs_op(lhs_lt, op, py_numeric_scalar_cast<T>(args[1]));
        return s(expr, opstr, exhw);
      }

      throw py::value_error("Unsupported local tensor scheduler operation. Currently supported: "
                            "(local_lhs, '=', scalar), '+= scalar', and '-= scalar'.");
    },
    py::arg("lhs"), py::return_value_policy::reference_internal);
}

// -----------------------------------------------------------------------------
// Tensor utility bindings
// -----------------------------------------------------------------------------

template<typename T>
void bind_tamm_utils_for_type(py::module_& m) {
  using PyTensorT  = PyTensor<T>;
  using PyLabeledT = PyLabeledTensor<T>;

  m.def(
    "write_to_disk",
    [](std::shared_ptr<PyTensorT> tensor, const std::string& filename, bool tammio, bool profile,
       int nagg_hint) {
      if(!tensor) throw py::value_error("write_to_disk: tensor must not be None");
      Tensor<T>              raw = raw_tensor(tensor);
      py::gil_scoped_release release;
      tamm::write_to_disk<T>(raw, filename, tammio, profile, nagg_hint);
    },
    py::arg("tensor"), py::arg("filename"), py::arg("tammio") = true, py::arg("profile") = false,
    py::arg("nagg_hint") = 0);

  m.def(
    "read_from_disk",
    [](std::shared_ptr<PyTensorT> tensor, const std::string& filename, bool tammio,
       py::object wtensor_obj, bool profile, int nagg_hint) {
      if(!tensor) throw py::value_error("read_from_disk: tensor must not be None");

      Tensor<T> raw     = raw_tensor(tensor);
      Tensor<T> wtensor = {};

      if(!wtensor_obj.is_none()) {
        auto wtensor_py = wtensor_obj.cast<std::shared_ptr<PyTensorT>>();
        if(!wtensor_py) throw py::value_error("read_from_disk: wtensor must not be None");
        wtensor = wtensor_py->raw();
      }

      py::gil_scoped_release release;
      tamm::read_from_disk<T>(raw, filename, tammio, wtensor, profile, nagg_hint);
    },
    py::arg("tensor"), py::arg("filename"), py::arg("tammio") = true,
    py::arg("wtensor") = py::none(), py::arg("profile") = false, py::arg("nagg_hint") = 0);

  m.def(
    "get_scalar",
    [](PyTensorT& tensor) { return py_scalar_from_value(tamm::get_scalar<T>(tensor.raw())); },
    py::arg("tensor"));

  m.def(
    "identity_matrix",
    [](ExecutionContext& ec, const TiledIndexSpace& tis) {
      return wrap_tensor<T>(tamm::identity_matrix<T>(ec, tis));
    },
    py::arg("ec"), py::arg("tis"));

  m.def(
    "random_ip",
    [](std::shared_ptr<PyLabeledT> t, unsigned int seed) {
      tamm::random_ip<T>(raw_labeled(t), seed);
    },
    py::arg("tensor"), py::arg("seed") = 0);

  m.def(
    "random_ip",
    [](std::shared_ptr<PyTensorT> t, unsigned int seed) {
      tamm::random_ip<T>(raw_tensor(t), seed);
    },
    py::arg("tensor"), py::arg("seed") = 0);

#define TAMM_BIND_REDUCTION(NAME)                                 \
  m.def(                                                          \
    #NAME,                                                        \
    [](std::shared_ptr<PyTensorT> t) {                            \
      return py_scalar_from_value(tamm::NAME<T>(raw_tensor(t)));  \
    },                                                            \
    py::arg("tensor"));                                           \
  m.def(                                                          \
    #NAME,                                                        \
    [](std::shared_ptr<PyLabeledT> t) {                           \
      return py_scalar_from_value(tamm::NAME<T>(raw_labeled(t))); \
    },                                                            \
    py::arg("tensor"))

  TAMM_BIND_REDUCTION(trace);
  TAMM_BIND_REDUCTION(trace_sqr);
  TAMM_BIND_REDUCTION(norm);
  TAMM_BIND_REDUCTION(linf_norm);
  TAMM_BIND_REDUCTION(sum);

#undef TAMM_BIND_REDUCTION

  m.def(
    "diagonal", [](std::shared_ptr<PyTensorT> t) { return tamm::diagonal<T>(raw_tensor(t)); },
    py::arg("tensor"));
  m.def(
    "diagonal", [](std::shared_ptr<PyLabeledT> t) { return tamm::diagonal<T>(raw_labeled(t)); },
    py::arg("tensor"));

  m.def(
    "update_diagonal",
    [](std::shared_ptr<PyTensorT> t, const std::vector<T>& d) {
      tamm::update_diagonal<T>(raw_tensor(t), d);
    },
    py::arg("tensor"), py::arg("diag"));

  m.def(
    "update_diagonal",
    [](std::shared_ptr<PyLabeledT> t, const std::vector<T>& d) {
      tamm::update_diagonal<T>(raw_labeled(t), d);
    },
    py::arg("tensor"), py::arg("diag"));

#define TAMM_BIND_UNARY(NAME)                                                                  \
  m.def(                                                                                       \
    #NAME,                                                                                     \
    [](std::shared_ptr<PyTensorT> t) { return wrap_tensor<T>(tamm::NAME<T>(raw_tensor(t))); }, \
    py::arg("tensor"));                                                                        \
  m.def(                                                                                       \
    #NAME,                                                                                     \
    [](std::shared_ptr<PyLabeledT> t, bool is_lt) {                                            \
      return wrap_tensor<T>(tamm::NAME<T>(raw_labeled(t), is_lt));                             \
    },                                                                                         \
    py::arg("tensor"), py::arg("is_lt") = true)

  TAMM_BIND_UNARY(sqrt);
  TAMM_BIND_UNARY(square);
  TAMM_BIND_UNARY(log);
  TAMM_BIND_UNARY(log10);
  TAMM_BIND_UNARY(einverse);

#undef TAMM_BIND_UNARY

  if constexpr(std::is_same_v<T, double>) {
    m.def(
      "print_memory_usage",
      [](int64_t rank) { tamm::print_memory_usage<double>(static_cast<int>(rank)); },
      py::arg("rank"));
  }

  if constexpr(tamm::internal::is_complex_v<T>) {
    m.def(
      "conj",
      [](std::shared_ptr<PyTensorT> t) { return wrap_tensor<T>(tamm::conj<T>(raw_tensor(t))); },
      py::arg("tensor"));

    m.def(
      "conj",
      [](std::shared_ptr<PyLabeledT> t, bool is_lt) {
        return wrap_tensor<T>(tamm::conj<T>(raw_labeled(t), is_lt));
      },
      py::arg("tensor"), py::arg("is_lt") = true);

    m.def(
      "conj_ip", [](std::shared_ptr<PyTensorT> t) { tamm::conj_ip<T>(raw_tensor(t)); },
      py::arg("tensor"));

    m.def(
      "conj_ip", [](std::shared_ptr<PyLabeledT> t) { tamm::conj_ip<T>(raw_labeled(t)); },
      py::arg("tensor"));
  }

  m.def(
    "scale",
    [](std::shared_ptr<PyTensorT> t, T alpha) {
      return wrap_tensor<T>(tamm::scale<T>(raw_tensor(t), alpha));
    },
    py::arg("tensor"), py::arg("alpha"));

  m.def(
    "scale",
    [](std::shared_ptr<PyLabeledT> t, T alpha, bool is_lt) {
      return wrap_tensor<T>(tamm::scale<T>(raw_labeled(t), alpha, is_lt));
    },
    py::arg("tensor"), py::arg("alpha"), py::arg("is_lt") = true);

  m.def(
    "scale_ip",
    [](std::shared_ptr<PyTensorT> t, T alpha) { tamm::scale_ip<T>(raw_tensor(t), alpha); },
    py::arg("tensor"), py::arg("alpha"));

  m.def(
    "scale_ip",
    [](std::shared_ptr<PyLabeledT> t, T alpha) { tamm::scale_ip<T>(raw_labeled(t), alpha); },
    py::arg("tensor"), py::arg("alpha"));

  if constexpr(std::is_same_v<T, double>) {
    m.def(
      "max_element",
      [](std::shared_ptr<PyTensorT> t) { return tamm::max_element<T>(raw_tensor(t)); },
      py::arg("tensor"));
    m.def(
      "max_element",
      [](std::shared_ptr<PyLabeledT> t) { return tamm::max_element<T>(raw_labeled(t)); },
      py::arg("tensor"));
    m.def(
      "min_element",
      [](std::shared_ptr<PyTensorT> t) { return tamm::min_element<T>(raw_tensor(t)); },
      py::arg("tensor"));
    m.def(
      "min_element",
      [](std::shared_ptr<PyLabeledT> t) { return tamm::min_element<T>(raw_labeled(t)); },
      py::arg("tensor"));
  }

  m.def(
    "update_tensor_val",
    [](ExecutionContext& ec, std::shared_ptr<PyLabeledT> t, std::vector<size_t> coord, T value) {
      tamm::update_tensor_val<T>(ec, raw_labeled(t), coord, value);
    },
    py::arg("ec"), py::arg("tensor"), py::arg("coord"), py::arg("value"));

  m.def(
    "update_tensor_val",
    [](std::shared_ptr<PyLabeledT> t, std::vector<size_t> coord, T value) {
      tamm::update_tensor_val<T>(raw_labeled(t), coord, value);
    },
    py::arg("tensor"), py::arg("coord"), py::arg("value"));

  m.def(
    "update_tensor_val",
    [](ExecutionContext& ec, std::shared_ptr<PyTensorT> t, std::vector<size_t> coord, T value) {
      tamm::update_tensor_val<T>(ec, raw_tensor(t), coord, value);
    },
    py::arg("ec"), py::arg("tensor"), py::arg("coord"), py::arg("value"));

  m.def(
    "update_tensor_val",
    [](std::shared_ptr<PyTensorT> t, std::vector<size_t> coord, T value) {
      tamm::update_tensor_val<T>(raw_tensor(t), coord, value);
    },
    py::arg("tensor"), py::arg("coord"), py::arg("value"));

  m.def(
    "hash_tensor", [](std::shared_ptr<PyTensorT> t) { return tamm::hash_tensor<T>(raw_tensor(t)); },
    py::arg("tensor"));

  m.def(
    "permute_tensor",
    [](std::shared_ptr<PyTensorT> t, std::vector<int> permute) {
      return wrap_tensor<T>(tamm::permute_tensor<T>(raw_tensor(t), permute));
    },
    py::arg("tensor"), py::arg("permute"));

  m.def(
    "to_dense_tensor",
    [](ExecutionContext& ec_dense, std::shared_ptr<PyTensorT> t) {
      return wrap_tensor<T>(tamm::to_dense_tensor<T>(ec_dense, raw_tensor(t)));
    },
    py::arg("ec_dense"), py::arg("tensor"));

  m.def(
    "get_tensor_element",
    [](std::shared_ptr<PyTensorT> t, std::vector<int64_t> index_id) {
      return py_scalar_from_value(tamm::get_tensor_element<T>(raw_tensor(t), index_id));
    },
    py::arg("tensor"), py::arg("index_id"));

  m.def(
    "tensor_block",
    [](std::shared_ptr<PyTensorT> t, std::vector<int64_t> lo, std::vector<int64_t> hi,
       std::vector<int> permute) {
      return wrap_tensor<T>(tamm::tensor_block<T>(raw_tensor(t), lo, hi, permute));
    },
    py::arg("tensor"), py::arg("lo"), py::arg("hi"), py::arg("permute") = std::vector<int>{});

  m.def(
    "redistribute_tensor",
    [](ExecutionContext& ec, std::shared_ptr<PyTensorT> t, TiledIndexSpaceVec tis,
       std::vector<size_t> spins) {
      return wrap_tensor<T>(tamm::redistribute_tensor<T>(ec, raw_tensor(t), tis, spins));
    },
    py::arg("ec"), py::arg("tensor"), py::arg("tis"), py::arg("spins") = std::vector<size_t>{});

  m.def(
    "redistribute_tensor",
    [](std::shared_ptr<PyTensorT> t, TiledIndexSpaceVec tis, std::vector<size_t> spins) {
      return wrap_tensor<T>(tamm::redistribute_tensor<T>(raw_tensor(t), tis, spins));
    },
    py::arg("tensor"), py::arg("tis"), py::arg("spins") = std::vector<size_t>{});

  m.def(
    "retile_tamm_tensor",
    [](std::shared_ptr<PyTensorT> stensor, std::shared_ptr<PyTensorT> dtensor, std::string name) {
      tamm::retile_tamm_tensor<T>(raw_tensor(stensor), raw_tensor(dtensor), name);
    },
    py::arg("stensor"), py::arg("dtensor"), py::arg("tname") = "");

  m.def(
    "pow",
    [](std::shared_ptr<PyTensorT> t, T alpha) {
      return wrap_tensor<T>(tamm::pow<T>(raw_tensor(t), alpha));
    },
    py::arg("tensor"), py::arg("alpha"));

  m.def(
    "pow",
    [](std::shared_ptr<PyLabeledT> t, T alpha, bool is_lt) {
      return wrap_tensor<T>(tamm::pow<T>(raw_labeled(t), alpha, is_lt));
    },
    py::arg("tensor"), py::arg("alpha"), py::arg("is_lt") = true);

  m.def(
    "print_tensor",
    [](std::shared_ptr<PyLabeledT> t, const std::string& filename) {
      auto lt = raw_labeled(t);
      tamm::print_tensor<T>(lt.tensor(), filename);
    },
    py::arg("tensor"), py::arg("filename") = "");

  m.def(
    "print_tensor_all",
    [](std::shared_ptr<PyTensorT> t, const std::string& filename) {
      tamm::print_tensor_all<T>(raw_tensor(t), filename);
    },
    py::arg("tensor"), py::arg("filename") = "");

  m.def(
    "print_tensor_all",
    [](std::shared_ptr<PyLabeledT> t, const std::string& filename) {
      auto lt = raw_labeled(t);
      tamm::print_tensor_all<T>(lt.tensor(), filename);
    },
    py::arg("tensor"), py::arg("filename") = "");

  m.def(
    "print_tensor_reshaped",
    [](std::shared_ptr<PyLabeledT> t, const IndexLabelVec& new_labels) {
      tamm::print_tensor_reshaped<T>(raw_labeled(t), new_labels);
    },
    py::arg("tensor"), py::arg("new_labels"));

  m.def(
    "print_labeled_tensor",
    [](std::shared_ptr<PyLabeledT> t) { tamm::print_labeled_tensor<T>(raw_labeled(t)); },
    py::arg("tensor"));

  m.def(
    "print_vector",
    [](const std::vector<T>& vec, const std::string& filename) {
      tamm::print_vector<T>(vec, filename);
    },
    py::arg("vec"), py::arg("filename") = "");

  m.def(
    "print_max_above_threshold",
    [](std::shared_ptr<PyTensorT> t, double printtol, const std::string& filename) {
      tamm::print_max_above_threshold<T>(raw_tensor(t), printtol, filename);
    },
    py::arg("tensor"), py::arg("printtol"), py::arg("filename") = "");

  m.def(
    "print_dense_tensor",
    [](std::shared_ptr<PyTensorT> t, const std::string& filename, bool append) {
      std::function<bool(std::vector<size_t>)> all = [](std::vector<size_t>) { return true; };
      tamm::print_dense_tensor<T>(raw_tensor(t), all, filename, append);
    },
    py::arg("tensor"), py::arg("filename") = "", py::arg("append") = false);

  m.def(
    "print_dense_tensor",
    [](std::shared_ptr<PyTensorT> t, py::function predicate, const std::string& filename,
       bool append) {
      std::function<bool(std::vector<size_t>)> pred = [predicate](std::vector<size_t> coord) {
        py::gil_scoped_acquire gil;
        py::tuple              py_coord(coord.size());
        for(size_t i = 0; i < coord.size(); ++i) py_coord[i] = py::int_(coord[i]);
        return predicate(py_coord).cast<bool>();
      };

      tamm::print_dense_tensor<T>(raw_tensor(t), pred, filename, append);
    },
    py::arg("tensor"), py::arg("predicate"), py::arg("filename") = "", py::arg("append") = false);

  if constexpr(std::is_same_v<T, double>) {
    m.def("print_varlist", [](py::args args) {
      for(py::ssize_t i = 0; i < static_cast<py::ssize_t>(args.size()); ++i) {
        if(i != 0) std::cout << ",";
        std::cout << py::str(args[i]).cast<std::string>();
      }
      std::cout << std::endl;
    });

    m.def(
      "print_memory_usage",
      [](int64_t rank, const std::string& mstring, bool complex_elements) {
        if(complex_elements) tamm::print_memory_usage<std::complex<double>>(rank, mstring);
        else tamm::print_memory_usage<double>(rank, mstring);
      },
      py::arg("rank"), py::arg("mstring") = "", py::arg("complex_elements") = false);

    m.def(
      "print_memory_usage_double",
      [](int64_t rank, const std::string& mstring) {
        tamm::print_memory_usage<double>(rank, mstring);
      },
      py::arg("rank"), py::arg("mstring") = "");

    m.def(
      "print_memory_usage_complex_double",
      [](int64_t rank, const std::string& mstring) {
        tamm::print_memory_usage<std::complex<double>>(rank, mstring);
      },
      py::arg("rank"), py::arg("mstring") = "");
  }
}

// -----------------------------------------------------------------------------
// Tensor-family bindings
// -----------------------------------------------------------------------------

template<typename T>
void bind_tensor_family(py::module_& m, py::class_<Scheduler>& scheduler_cls) {
  using PyTensorT  = PyTensor<T>;
  using PyLabeledT = PyLabeledTensor<T>;
  using TensorT    = Tensor<T>;

  const std::string tensor_name = tensor_class_name<T>();
  const std::string lt_name     = labeled_tensor_class_name<T>();

  auto lt_cls = py::class_<PyLabeledT, std::shared_ptr<PyLabeledT>>(m, lt_name.c_str());

  lt_cls.def("labels", [](const PyLabeledT& lt) { return lt.labels; })
    .def("tensor", [](const PyLabeledT& lt) { return lt.tensor; })
    .def(
      "to_numpy",
      [](PyLabeledT& lt, py::object dtype) {
        return numpy_array_protocol_result(tamm_tensor_to_numpy_array<T>(lt.raw_tensor()), dtype);
      },
      py::arg("dtype") = py::none())
    .def(
      "__array__",
      [](PyLabeledT& lt, py::object dtype, py::object copy) {
        return numpy_array_protocol_result(tamm_tensor_to_numpy_array<T>(lt.raw_tensor()), dtype,
                                           copy);
      },
      py::arg("dtype") = py::none(), py::arg("copy") = py::none())
    .def(
      "__mul__",
      [](std::shared_ptr<PyLabeledT> lhs, py::object rhs) {
        auto lhs_expr = PyTensorExpr::from_tensor(lhs);
        if(py_is_numeric_scalar(rhs)) return PyTensorExpr::scaled(py_to_expr_scalar(rhs), lhs_expr);
        return PyTensorExpr::mul(lhs_expr, py_to_tensor_expr(rhs));
      },
      py::is_operator())
    .def(
      "__rmul__",
      [](std::shared_ptr<PyLabeledT> rhs, py::object lhs) {
        if(!py_is_numeric_scalar(lhs)) {
          throw py::type_error("Only scalar * labeled tensor is supported");
        }
        return PyTensorExpr::scaled(py_to_expr_scalar(lhs), PyTensorExpr::from_tensor(rhs));
      },
      py::is_operator())
    .def("assign_expr",
         [](std::shared_ptr<PyLabeledT> lhs, const PyTensorExpr& expr) {
           if(!lhs) throw py::value_error("lhs must not be None");
           return make_py_op_expr_from_expr(*lhs, "=", expr);
         })
    .def("add_expr",
         [](std::shared_ptr<PyLabeledT> lhs, const PyTensorExpr& expr) {
           if(!lhs) throw py::value_error("lhs must not be None");
           return make_py_op_expr_from_expr(*lhs, "+=", expr);
         })
    .def("sub_expr", [](std::shared_ptr<PyLabeledT> lhs, const PyTensorExpr& expr) {
      if(!lhs) throw py::value_error("lhs must not be None");
      return make_py_op_expr_from_expr(*lhs, "-=", expr);
    });

  if constexpr(std::is_same_v<T, double>) {
    lt_cls
      .def("as_op",
           [](std::shared_ptr<PyLabeledT> self) -> std::shared_ptr<NewOpBase> {
             return make_ltop_expr_from_labeled_double(self);
           })
      .def(
        "update",
        [](std::shared_ptr<PyLabeledT>       self,
           const std::shared_ptr<NewOpBase>& rhs) -> std::shared_ptr<PyLabeledT> {
          update_labeled_double_with_new_op(self, rhs);
          return self;
        },
        py::arg("rhs"));
  }

  py::class_<PyTensorT, std::shared_ptr<PyTensorT>>(m, tensor_name.c_str())
    .def(py::init([]() { return std::make_shared<PyTensorT>(); }))
    .def(py::init([](py::sequence seq) {
      return std::make_shared<PyTensorT>(make_tensor_from_sequence<T>(seq));
    }))
    .def(py::init([](TiledIndexSpaceVec tis_vec, SpinMask spin_mask) {
      return std::make_shared<PyTensorT>(TensorT{tis_vec, spin_mask});
    }))
    .def(py::init([](IndexLabelVec labels, SpinMask spin_mask) {
      return std::make_shared<PyTensorT>(TensorT{labels, spin_mask});
    }))
    .def(py::init([](TiledIndexSpaceVec tis_vec, std::vector<size_t> spins) {
      return std::make_shared<PyTensorT>(TensorT{tis_vec, spins});
    }))
    .def(py::init([](IndexLabelVec labels, std::vector<size_t> spins) {
      return std::make_shared<PyTensorT>(TensorT{labels, spins});
    }))
    .def(py::init([](const TiledIndexSpaceVec& tis_vec, const TensorInfo& info) {
      return std::make_shared<PyTensorT>(TensorT{tis_vec, info});
    }))
    .def(py::init([](TiledIndexSpaceVec tis_vec, const std::vector<std::string>& allowed,
                     const Char2TISMap& char_map) {
      return std::make_shared<PyTensorT>(TensorT{tis_vec, allowed, char_map});
    }))
    .def(py::init([](TiledIndexSpaceVec tis_vec, const std::vector<std::string>& allowed) {
      return std::make_shared<PyTensorT>(TensorT{tis_vec, allowed});
    }))
    .def(py::init([](TiledIndexSpaceVec tis_vec, const std::vector<TiledIndexSpaceVec>& allowed) {
      return std::make_shared<PyTensorT>(TensorT{tis_vec, allowed});
    }))
    .def(py::init([](TiledIndexSpaceVec tis_vec, const std::vector<IndexLabelVec>& allowed) {
      return std::make_shared<PyTensorT>(TensorT{tis_vec, allowed});
    }))
    .def(py::init([](py::sequence seq, py::function pyfunc) {
           using Func = std::function<void(const IndexVector&, span<T>)>;

           bool all_tis = true;
           bool all_lbl = true;

           for(auto item: seq) {
             py::handle h = item;
             all_tis      = all_tis && py::isinstance<TiledIndexSpace>(h);
             all_lbl      = all_lbl && py::isinstance<TiledIndexLabel>(h);
           }

           Func f = [pyfunc](const IndexVector& blockid, span<T> buff) {
             py::gil_scoped_acquire gil;

             auto py_blockid = from_index_vector(blockid);
             auto py_buff    = py::array_t<T>({static_cast<py::ssize_t>(buff.size())},
                                           {static_cast<py::ssize_t>(sizeof(T))}, buff.data(),
                                           py::none());

             pyfunc(py_blockid, py_buff);
           };

           if(all_tis) {
             TiledIndexSpaceVec spaces;
             spaces.reserve(static_cast<size_t>(py::len(seq)));
             for(auto item: seq) spaces.push_back(item.cast<TiledIndexSpace>());
             return std::make_shared<PyTensorT>(Tensor<T>{spaces, f});
           }

           if(all_lbl) {
             IndexLabelVec labels;
             labels.reserve(static_cast<size_t>(py::len(seq)));
             for(auto item: seq) labels.push_back(item.cast<TiledIndexLabel>());
             return std::make_shared<PyTensorT>(Tensor<T>{labels, f});
           }

           throw py::type_error("Tensor(seq, callable) expects seq to contain all "
                                "TiledIndexSpace or all TiledIndexLabel objects");
         }),
         py::arg("spaces_or_labels"), py::arg("func"))
    .def(py::init([](std::shared_ptr<PyTensorT> src, size_t unit_tiled_ndims) {
           if(!src) throw py::value_error("Source tensor must not be None");
           return std::make_shared<PyTensorT>(TensorT{src->raw(), unit_tiled_ndims}, src);
         }),
         py::arg("source_tensor"), py::arg("unit_tiled_ndims"))
    .def_static(
      "allocate",
      [](ExecutionContext& ec, py::args args) {
        for(auto obj: args) {
          auto t = obj.cast<std::shared_ptr<PyTensorT>>();
          if(!t) throw py::value_error("Tensor.allocate received None");
          t->allocate(ec);
        }
      },
      py::arg("ec"), py::keep_alive<0, 1>())
    .def(
      "allocate_self", [](PyTensorT& t, ExecutionContext& ec) { t.allocate(ec); }, py::arg("ec"))
    .def(
      "allocate_force", [](PyTensorT& t, ExecutionContext& ec) { t.allocate_force(ec); },
      py::arg("ec"))
    .def(
      "same_memory_region",
      [](PyTensorT& self, std::shared_ptr<PyTensorT> other) {
        if(!other) throw py::value_error("other tensor must not be None");
        return self.same_memory_region(*other);
      },
      py::arg("other"))
    .def(
      "same_distribution",
      [](PyTensorT& self, std::shared_ptr<PyTensorT> other) {
        if(!other) throw py::value_error("other tensor must not be None");
        return self.same_distribution(*other);
      },
      py::arg("other"))
    .def("deallocate", [](PyTensorT& t) { t.deallocate(); })
    .def("is_allocated", &PyTensorT::is_allocated)
    .def("__call__",
         [](PyTensorT& self, py::args args) {
           return make_labeled_tensor_from_args<T>(self.shared_from_this(), args);
         })
    .def(
      "block_size",
      [](PyTensorT& t, py::sequence seq) { return t.raw().block_size(to_index_vector(seq)); },
      py::arg("blockid"))
    .def(
      "block_dims",
      [](PyTensorT& t, py::sequence seq) { return t.raw().block_dims(to_index_vector(seq)); },
      py::arg("blockid"))
    .def(
      "block_offsets",
      [](PyTensorT& t, py::sequence seq) { return t.raw().block_offsets(to_index_vector(seq)); },
      py::arg("blockid"))
    .def(
      "get",
      [](PyTensorT& t, py::sequence seq, py::list pybuf) {
        IndexVector idx_vec = to_index_vector(seq);
        const auto  bs      = t.raw().block_size(idx_vec);

        if(py::len(pybuf) != bs) {
          throw std::runtime_error("get: Python list size does not match tensor block size");
        }

        std::vector<T> buf(bs);
        gsl::span<T>   sp(buf.data(), buf.size());
        t.raw().get(idx_vec, sp);

        for(size_t i = 0; i < bs; ++i) pybuf[i] = py_scalar_from_value(buf[i]);
      },
      py::arg("blockid"), py::arg("buf"))
    .def(
      "is_non_zero",
      [](PyTensorT& t, py::sequence seq) { return t.raw().is_non_zero(to_index_vector(seq)); },
      py::arg("blockid"))
    .def("loop_nest", [](PyTensorT& t) { return t.raw().loop_nest(); })
    .def("tiled_index_spaces", [](PyTensorT& t) { return t.raw().tiled_index_spaces(); })
    .def(
      "execution_context", [](PyTensorT& t) { return t.raw().execution_context(); },
      py::return_value_policy::reference_internal)
    .def("num_modes", [](PyTensorT& t) { return t.raw().num_modes(); })
    .def("local_buf_size", [](PyTensorT& t) { return t.raw().local_buf_size(); })
    .def("size", [](PyTensorT& t) { return t.raw().size(); })
    .def(
      "set_dense", [](PyTensorT& t, ProcGrid pg) { t.raw().set_dense(pg); },
      py::arg("pg") = ProcGrid{})
    .def("set_block_cyclic", [](PyTensorT& t, ProcGrid pg) { t.raw().set_block_cyclic(pg); })
    .def("set_restricted", [](PyTensorT& t, ProcList pl) { t.raw().set_restricted(pl); })
    .def("is_dense", [](PyTensorT& t) { return t.raw().is_dense(); })
    .def("is_sparse", [](PyTensorT& t) { return t.raw().is_sparse(); })
    .def("is_block_cyclic", [](PyTensorT& t) { return t.raw().is_block_cyclic(); })
    .def(
      "access_local_buf",
      [](PyTensorT& t) -> py::object {
        auto* ptr = t.raw().access_local_buf();
        if(ptr == nullptr) return py::none();

        return py::array_t<T>({static_cast<py::ssize_t>(t.raw().local_buf_size())},
                              {static_cast<py::ssize_t>(sizeof(T))}, ptr, py::cast(&t));
      },
      py::return_value_policy::reference_internal)
    .def(
      "to_numpy",
      [](PyTensorT& t, py::object dtype) {
        return numpy_array_protocol_result(tamm_tensor_to_numpy_array<T>(t.raw()), dtype);
      },
      py::arg("dtype") = py::none())
    .def(
      "__array__",
      [](PyTensorT& t, py::object dtype, py::object copy) {
        return numpy_array_protocol_result(tamm_tensor_to_numpy_array<T>(t.raw()), dtype, copy);
      },
      py::arg("dtype") = py::none(), py::arg("copy") = py::none());

  m.def(
    "to_numpy",
    [](std::shared_ptr<PyTensorT> t, py::object dtype) {
      if(!t) throw py::value_error("to_numpy: Tensor must not be None");
      return numpy_array_protocol_result(tamm_tensor_to_numpy_array<T>(t->raw()), dtype);
    },
    py::arg("tensor"), py::arg("dtype") = py::none());

  m.def(
    "to_numpy",
    [](std::shared_ptr<PyLabeledT> t, py::object dtype) {
      if(!t) throw py::value_error("to_numpy: LabeledTensor must not be None");
      return numpy_array_protocol_result(tamm_tensor_to_numpy_array<T>(t->raw_tensor()), dtype);
    },
    py::arg("tensor"), py::arg("dtype") = py::none());

  m.def(
    "assign",
    [](std::shared_ptr<PyLabeledT> t, py::object value) {
      if(!t) throw py::value_error("assign: tensor must not be None");
      if(!py_is_scalar_compatible<T>(value)) throw py::type_error("Incompatible scalar");
      return make_py_op_expr_from_scalar(*t, "=", py_numeric_scalar_cast<T>(value));
    },
    py::arg("tensor"), py::arg("value"));

  m.def(
    "assign",
    [](std::shared_ptr<PyTensorT> t, py::object value) {
      if(!t) throw py::value_error("assign: tensor must not be None");
      if(!py_is_scalar_compatible<T>(value)) throw py::type_error("Incompatible scalar");

      auto       lt = t->raw()();
      PyLabeledT proxy{t, lt.labels()};
      return make_py_op_expr_from_scalar(proxy, "=", py_numeric_scalar_cast<T>(value));
    },
    py::arg("tensor"), py::arg("value"));

  m.def(
    "expr", [](std::shared_ptr<PyLabeledT> lt) { return PyTensorExpr::from_tensor(lt); },
    py::arg("tensor"));

  m.def(
    "assign_expr",
    [](std::shared_ptr<PyLabeledT> lhs, const PyTensorExpr& expr) {
      if(!lhs) throw py::value_error("lhs must not be None");
      return make_py_op_expr_from_expr(*lhs, "=", expr);
    },
    py::arg("lhs"), py::arg("expr"));

  m.def(
    "add_expr",
    [](std::shared_ptr<PyLabeledT> lhs, const PyTensorExpr& expr) {
      if(!lhs) throw py::value_error("lhs must not be None");
      return make_py_op_expr_from_expr(*lhs, "+=", expr);
    },
    py::arg("lhs"), py::arg("expr"));

  m.def(
    "sub_expr",
    [](std::shared_ptr<PyLabeledT> lhs, const PyTensorExpr& expr) {
      if(!lhs) throw py::value_error("lhs must not be None");
      return make_py_op_expr_from_expr(*lhs, "-=", expr);
    },
    py::arg("lhs"), py::arg("expr"));

  m.def(
    "copy",
    [](std::shared_ptr<PyLabeledT> lhs, py::object rhs) {
      if(!lhs) throw py::value_error("lhs must not be None");
      return make_py_op_expr_from_expr(*lhs, "=", py_to_tensor_expr(rhs));
    },
    py::arg("t1"), py::arg("t2"));

  m.def(
    "copy_add",
    [](std::shared_ptr<PyLabeledT> lhs, py::object rhs) {
      if(!lhs) throw py::value_error("lhs must not be None");
      return make_py_op_expr_from_expr(*lhs, "+=", py_to_tensor_expr(rhs));
    },
    py::arg("t1"), py::arg("t2"));

  m.def(
    "copy_sub",
    [](std::shared_ptr<PyLabeledT> lhs, py::object rhs) {
      if(!lhs) throw py::value_error("lhs must not be None");
      return make_py_op_expr_from_expr(*lhs, "-=", py_to_tensor_expr(rhs));
    },
    py::arg("t1"), py::arg("t2"));

  m.def("multiply_lt_lt", [](std::shared_ptr<PyLabeledT> lhs, py::object A, py::object B) {
    if(!lhs) throw py::value_error("lhs must not be None");
    return make_py_op_expr_from_expr(*lhs, "=",
                                     PyTensorExpr::mul(py_to_tensor_expr(A), py_to_tensor_expr(B)));
  });

  m.def("add_multiply_lt_lt", [](std::shared_ptr<PyLabeledT> lhs, py::object A, py::object B) {
    if(!lhs) throw py::value_error("lhs must not be None");
    return make_py_op_expr_from_expr(
      *lhs, "+=", PyTensorExpr::mul(py_to_tensor_expr(A), py_to_tensor_expr(B)));
  });

  m.def("add_multiply_alpha_lt_lt", [](std::shared_ptr<PyLabeledT> lhs, py::object alpha,
                                       py::object A, py::object B) {
    if(!lhs) throw py::value_error("lhs must not be None");
    if(!py_is_numeric_scalar(alpha)) throw py::type_error("alpha must be numeric");

    auto expr = PyTensorExpr::scaled(py_to_expr_scalar(alpha),
                                     PyTensorExpr::mul(py_to_tensor_expr(A), py_to_tensor_expr(B)));

    return make_py_op_expr_from_expr(*lhs, "+=", expr);
  });

  m.def("add_multiply_alpha_lt",
        [](std::shared_ptr<PyLabeledT> lhs, py::object alpha, py::object A) {
          if(!lhs) throw py::value_error("lhs must not be None");
          if(!py_is_numeric_scalar(alpha)) throw py::type_error("alpha must be numeric");

          auto expr = PyTensorExpr::scaled(py_to_expr_scalar(alpha), py_to_tensor_expr(A));
          return make_py_op_expr_from_expr(*lhs, "+=", expr);
        });

  m.def(
    "translate_blockid",
    [](py::sequence seq, std::shared_ptr<PyLabeledT> ltensor) {
      if(!ltensor) throw py::value_error("translate_blockid: ltensor must not be None");

      auto in     = to_index_vector(seq);
      auto raw_lt = ltensor->materialize();
      auto out    = internal::translate_blockid<LabeledTensor<T>>(in, raw_lt);

      return from_index_vector(out);
    },
    py::arg("blockid"), py::arg("ltensor"));

  m.def(
    "print_tensor",
    [](std::shared_ptr<PyTensorT> t, std::string filename) {
      tamm::print_tensor<T>(raw_tensor(t), filename);
    },
    py::arg("tensor"), py::arg("filename") = "");

  m.def(
    "update_tensor",
    [](std::shared_ptr<PyLabeledT> tensor, py::function func) {
      if(!tensor) throw py::value_error("update_tensor: tensor must not be None");

      tamm::update_tensor<T>(raw_labeled(tensor), [func](const auto& blockid, auto&& buf) {
        py::gil_scoped_acquire gil;

        auto py_buf = py::array_t<T>({static_cast<py::ssize_t>(buf.size())},
                                     {static_cast<py::ssize_t>(sizeof(T))}, buf.data(), py::none());

        func(from_index_vector(blockid), py_buf);
      });
    },
    py::arg("tensor"), py::arg("func"));

  m.def(
    PyTypeInfo<T>::fill_name,
    [](std::shared_ptr<PyTensorT> tensor, py::function func) {
      tamm::fill_sparse_tensor<T>(raw_tensor(tensor), [func](const IndexVector& blockid,
                                                             span<T>            buf) {
        py::gil_scoped_acquire gil;

        auto py_buf = py::array_t<T>({static_cast<py::ssize_t>(buf.size())},
                                     {static_cast<py::ssize_t>(sizeof(T))}, buf.data(), py::none());

        func(from_index_vector(blockid), py_buf);
      });
    },
    py::arg("tensor"), py::arg("func"));

  m.def(
    PyTypeInfo<T>::fill_name,
    [](std::shared_ptr<PyLabeledT> tensor, py::function func) {
      tamm::fill_sparse_tensor<T>(raw_labeled(tensor), [func](const IndexVector& blockid,
                                                              span<T>            buf) {
        py::gil_scoped_acquire gil;

        auto py_buf = py::array_t<T>({static_cast<py::ssize_t>(buf.size())},
                                     {static_cast<py::ssize_t>(sizeof(T))}, buf.data(), py::none());

        func(from_index_vector(blockid), py_buf);
      });
    },
    py::arg("tensor"), py::arg("func"));

  scheduler_cls.def(
    "__call__",
    [](Scheduler& s, std::shared_ptr<PyLabeledT> lhs, py::args args,
       py::kwargs kwargs) -> Scheduler& {
      if(!lhs) throw py::value_error("lhs must not be None");

      std::string opstr;
      ExecutionHW exhw = ExecutionHW::DEFAULT;

      if(kwargs) {
        if(kwargs.contains("opstr")) opstr = kwargs["opstr"].cast<std::string>();
        if(kwargs.contains("exhw")) exhw = kwargs["exhw"].cast<ExecutionHW>();
      }

      PyOpExpr expr = make_py_op_expr_from_spec<T>(*lhs, args);
      return s(expr, opstr, exhw);
    },
    py::arg("lhs"), py::return_value_policy::reference_internal);

  scheduler_cls.def(
    "gop_copy",
    [](Scheduler& s, std::shared_ptr<PyLabeledT> lhs,
       std::shared_ptr<PyLabeledT> rhs) -> Scheduler& {
      if(!lhs) throw py::value_error("gop_copy lhs must not be None");
      if(!rhs) throw py::value_error("gop_copy rhs must not be None");

      auto lhs_lt = lhs->materialize();
      auto rhs_lt = rhs->materialize();

      auto lambda = [](const Tensor<T>&, const IndexVector&, std::vector<T>& lhs_buf,
                       const IndexVector[], std::vector<T> rhs_buf[]) {
        std::copy(rhs_buf[0].begin(), rhs_buf[0].end(), lhs_buf.begin());
      };

      std::array<LabeledTensor<T>, 1> rhs_arr{rhs_lt};
      return s.gop(lhs_lt, rhs_arr, lambda);
    },
    py::arg("lhs"), py::arg("rhs"), py::return_value_policy::reference_internal);

  scheduler_cls.def(
    "gop_zero_scan",
    [](Scheduler& s, std::shared_ptr<PyLabeledT> lhs) -> Scheduler& {
      if(!lhs) throw py::value_error("gop_zero_scan lhs must not be None");

      auto lhs_lt = lhs->materialize();
      auto lambda = [](Tensor<T>&, const IndexVector&, std::vector<T>& buf) {
        for(auto& v: buf) v = T{};
      };

      return s.gop(lhs_lt, lambda);
    },
    py::arg("lhs"), py::return_value_policy::reference_internal);

  scheduler_cls.def(
    "gop_sum2",
    [](Scheduler& s, std::shared_ptr<PyLabeledT> lhs, std::shared_ptr<PyLabeledT> rhs1,
       std::shared_ptr<PyLabeledT> rhs2) -> Scheduler& {
      if(!lhs || !rhs1 || !rhs2) throw py::value_error("gop_sum2 tensors must not be None");

      auto lhs_lt  = lhs->materialize();
      auto rhs1_lt = rhs1->materialize();
      auto rhs2_lt = rhs2->materialize();

      auto lambda = [](const Tensor<T>&, const IndexVector&, std::vector<T>& lhs_buf,
                       const IndexVector[], std::vector<T> rhs_buf[]) {
        for(size_t i = 0; i < lhs_buf.size(); ++i) { lhs_buf[i] = rhs_buf[0][i] + rhs_buf[1][i]; }
      };

      std::array<LabeledTensor<T>, 2> rhs_arr{rhs1_lt, rhs2_lt};
      return s.gop(lhs_lt, rhs_arr, lambda);
    },
    py::arg("lhs"), py::arg("rhs1"), py::arg("rhs2"), py::return_value_policy::reference_internal);

  scheduler_cls.def(
    "exact_copy",
    [](Scheduler& s, std::shared_ptr<PyLabeledT> lhs, std::shared_ptr<PyLabeledT> rhs,
       bool update) -> Scheduler& {
      if(!lhs) throw py::value_error("exact_copy lhs must not be None");
      if(!rhs) throw py::value_error("exact_copy rhs must not be None");

      return s.exact_copy(lhs->materialize(), rhs->materialize(), update);
    },
    py::arg("lhs"), py::arg("rhs"), py::arg("update") = false,
    py::return_value_policy::reference_internal);
}

// -----------------------------------------------------------------------------
// Additional operator bindings
// -----------------------------------------------------------------------------

static void bind_new_ops(py::module_& m) {
  py::class_<NewOpBase, std::shared_ptr<NewOpBase>>(m, "ExprOp", py::multiple_inheritance())
    .def("clone", [](const NewOpBase& self) -> py::object { return clone_new_op_py(self); })
    .def("test_print", &NewOpBase::test_print)
    .def("coeff", [](const NewOpBase& self) { return tamm_scalar_to_pyobject(self.coeff()); })
    .def(
      "set_coeff",
      [](NewOpBase& self, py::object value) { self.set_coeff(py_to_tamm_scalar(value)); },
      py::arg("value"))
    .def(
      "__neg__",
      [](const NewOpBase& self) -> py::object { return scale_new_op_py(self, tamm::Scalar{-1.0}); },
      py::is_operator())
    .def(
      "__add__",
      [](const NewOpBase& lhs, const NewOpBase& rhs) -> std::shared_ptr<NewAddOp> {
        return make_add_expr_typed(lhs, rhs);
      },
      py::is_operator())
    .def(
      "__radd__",
      [](const NewOpBase& rhs, py::object lhs) -> py::object {
        if(py_scalar_is_zero(lhs)) return clone_new_op_py(rhs);
        throw py::type_error("Only 0 + ExprOp is supported");
      },
      py::is_operator())
    .def(
      "__sub__",
      [](const NewOpBase& lhs, const NewOpBase& rhs) -> std::shared_ptr<NewAddOp> {
        return make_sub_expr_typed(lhs, rhs);
      },
      py::is_operator())
    .def(
      "__rsub__",
      [](const NewOpBase& rhs, py::object lhs) -> py::object {
        if(py_scalar_is_zero(lhs)) return scale_new_op_py(rhs, tamm::Scalar{-1.0});
        throw py::type_error("Only 0 - ExprOp is supported");
      },
      py::is_operator())
    .def(
      "__mul__",
      [](const NewOpBase& lhs, const NewOpBase& rhs) -> std::shared_ptr<NewMultOp> {
        return make_mult_expr_typed(lhs, rhs);
      },
      py::is_operator())
    .def(
      "__rmul__",
      [](const NewOpBase& rhs, py::object lhs) -> py::object {
        if(!py_is_numeric_scalar(lhs)) throw py::type_error("Expected numeric scalar");
        return scale_new_op_py(rhs, py_to_tamm_scalar(lhs));
      },
      py::is_operator())
    .def(
      "__mul__",
      [](const NewOpBase& lhs, py::object rhs) -> py::object {
        if(!py_is_numeric_scalar(rhs)) throw py::type_error("Expected numeric scalar");
        return scale_new_op_py(lhs, py_to_tamm_scalar(rhs));
      },
      py::is_operator());

  py::class_<NewMultOp, NewOpBase, std::shared_ptr<NewMultOp>>(m, "MultOp",
                                                               py::multiple_inheritance())
    .def("lhs", [](NewMultOp& self) -> py::object { return clone_new_op_py(self.lhs()); })
    .def("rhs", [](NewMultOp& self) -> py::object { return clone_new_op_py(self.rhs()); });

  py::class_<NewAddOp, NewOpBase, std::shared_ptr<NewAddOp>>(m, "AddOp", py::multiple_inheritance())
    .def("lhs", [](NewAddOp& self) -> py::object { return clone_new_op_py(self.lhs()); })
    .def("rhs", [](NewAddOp& self) -> py::object { return clone_new_op_py(self.rhs()); });

  py::class_<NewLTOp, NewOpBase, std::shared_ptr<NewLTOp>>(m, "LTOp", py::multiple_inheritance())
    .def(py::init<>())
    .def(py::init([](std::shared_ptr<PLTd> lt) { return make_ltop_value_from_labeled_double(lt); }),
         py::arg("labeled_tensor"))
    .def(py::init([](std::shared_ptr<PTd> t, const IndexLabelVec& labels) {
           if(!t) throw py::value_error("TensorDouble must not be None");
           return NewLTOp{t->raw(), labels};
         }),
         py::arg("tensor"), py::arg("labels") = IndexLabelVec{})
    .def("tensor_type", &NewLTOp::tensor_type)
    .def("labels", &NewLTOp::labels, py::return_value_policy::reference_internal)
    .def("is_equal", &NewLTOp::is_equal);

  m.def(
    "as_op",
    [](std::shared_ptr<PLTd> lt) -> std::shared_ptr<NewLTOp> {
      return make_ltop_expr_from_labeled_double_typed(lt);
    },
    py::arg("labeled_tensor"));

  m.def(
    "as_op",
    [](std::shared_ptr<PTd> t, const IndexLabelVec& labels) -> std::shared_ptr<NewLTOp> {
      if(!t) throw py::value_error("TensorDouble must not be None");
      return std::make_shared<NewLTOp>(t->raw(), labels);
    },
    py::arg("tensor"), py::arg("labels") = IndexLabelVec{});

  m.def(
    "update",
    [](std::shared_ptr<PLTd> lhs, const std::shared_ptr<NewOpBase>& rhs) -> std::shared_ptr<PLTd> {
      return update_labeled_double_with_new_op(lhs, rhs);
    },
    py::arg("lhs"), py::arg("rhs"));
}

// -----------------------------------------------------------------------------
// IndexSpace/TIS helpers for helping to convert dictionaries to maps
// -----------------------------------------------------------------------------

static std::map<IndexVector, IndexSpace> py_to_indexspace_dependency_map(py::dict dep_map_py) {
  std::map<IndexVector, IndexSpace> dep_map;

  for(auto item: dep_map_py) {
    dep_map.emplace(to_index_vector(item.first), item.second.cast<IndexSpace>());
  }

  return dep_map;
}

static TiledIndexSpaceVec py_to_tis_vec_checked(py::list spaces_py, const std::string& where) {
  TiledIndexSpaceVec spaces;
  spaces.reserve(py::len(spaces_py));

  for(auto item: spaces_py) {
    if(!py::isinstance<TiledIndexSpace>(item)) {
      throw py::type_error(where + ": expected list of TiledIndexSpace objects");
    }
    spaces.push_back(item.cast<TiledIndexSpace>());
  }

  return spaces;
}

// -----------------------------------------------------------------------------
// Intermediary method to register Symbols
// -----------------------------------------------------------------------------

static void py_register_symbols_named(SymbolTable& symbol_table, py::kwargs kwargs) {
  for(auto item: kwargs) {
    const std::string name = py::cast<std::string>(item.first);
    py::handle        obj  = item.second;

    if(py::isinstance<PTd>(obj)) {
      auto t                                  = obj.cast<std::shared_ptr<PTd>>();
      symbol_table[t->raw().get_symbol_ptr()] = name;
    }
    else if(py::isinstance<PTz>(obj)) {
      auto t                                  = obj.cast<std::shared_ptr<PTz>>();
      symbol_table[t->raw().get_symbol_ptr()] = name;
    }
    else if(py::isinstance<TiledIndexLabel>(obj)) {
      auto lbl                           = obj.cast<TiledIndexLabel>();
      symbol_table[lbl.get_symbol_ptr()] = name;
    }
    else {
      throw py::type_error("register_symbols only supports TensorDouble, TensorComplexDouble, "
                           "and TiledIndexLabel");
    }
  }
}

// -----------------------------------------------------------------------------
// Scheduler helpers
// Binding-only methods to assist with allocating and deallocating tensors
// -----------------------------------------------------------------------------

static Scheduler& scheduler_alloc_or_dealloc(Scheduler& s, py::args args, bool do_allocate) {
  if(args.size() == 0) return do_allocate ? s.allocate() : s.deallocate();

  auto apply = [&](auto& tensor) {
    if(do_allocate) s.allocate(tensor);
    else s.deallocate(tensor);
  };

  for(auto obj: args) {
    if(py::isinstance<PTd>(obj)) {
      auto t = obj.cast<std::shared_ptr<PTd>>();
      if(!t) throw py::value_error("Scheduler allocation received None TensorDouble");
      apply(t->raw());
    }
    else if(py::isinstance<PTz>(obj)) {
      auto t = obj.cast<std::shared_ptr<PTz>>();
      if(!t) throw py::value_error("Scheduler allocation received None TensorComplexDouble");
      apply(t->raw());
    }
    else if(py::isinstance<PLocalTd>(obj)) {
      auto t = obj.cast<std::shared_ptr<PLocalTd>>();
      if(!t) throw py::value_error("Scheduler allocation received None LocalTensorDouble");
      apply(t->raw());
    }
    else if(py::isinstance<PLocalTz>(obj)) {
      auto t = obj.cast<std::shared_ptr<PLocalTz>>();
      if(!t) throw py::value_error("Scheduler allocation received None LocalTensorComplexDouble");
      apply(t->raw());
    }
    else {
      throw py::type_error("Scheduler.allocate/deallocate only supports TensorDouble, "
                           "TensorComplexDouble, LocalTensorDouble, and "
                           "LocalTensorComplexDouble");
    }
  }

  return s;
}

// -----------------------------------------------------------------------------
// Core non-template bindings
// -----------------------------------------------------------------------------

static void bind_basic_types(py::module_& m) {
  py::bind_vector<StringLabelVec>(m, "StringLabelVec");
  py::bind_vector<IndexLabelVec>(m, "IndexLabelVec");

  bind_strong_num<Irrep, uint32_t>(m, "Irrep");
  bind_strong_num<Spin, int>(m, "Spin");
  bind_strong_num<Spatial, uint32_t>(m, "Spatial");
  bind_strong_num<Offset, uint64_t>(m, "Offset");
  bind_strong_num<BlockIndex, uint64_t>(m, "BlockIndex");
  bind_strong_num<Proc, int64_t>(m, "Proc");
  bind_strong_num<Sign, int32_t>(m, "Sign");

  py::bind_vector<IntLabelVec>(m, "IntLabelVec");
  py::bind_vector<SizeVec>(m, "SizeVec");
  py::bind_vector<ProcGrid>(m, "ProcGrid");
  py::bind_vector<SpinMask>(m, "SpinMask");

  py::enum_<AllocationStatus>(m, "AllocationStatus")
    .value("invalid", AllocationStatus::invalid)
    .value("created", AllocationStatus::created)
    .value("attached", AllocationStatus::attached)
    .value("deallocated", AllocationStatus::deallocated)
    .value("orphaned", AllocationStatus::orphaned)
    .export_values();

  py::enum_<ElementType>(m, "ElementType")
    .value("invalid", ElementType::invalid)
    .value("single_precision", ElementType::single_precision)
    .value("double_precision", ElementType::double_precision)
    .value("single_complex", ElementType::single_complex)
    .value("double_complex", ElementType::double_complex)
    .export_values();

  py::enum_<DistributionKind>(m, "DistributionKind")
    .value("invalid", DistributionKind::invalid)
    .value("nw", DistributionKind::nw)
    .value("dense", DistributionKind::dense)
    .value("simple_round_robin", DistributionKind::simple_round_robin)
    .value("view", DistributionKind::view)
    .value("unit_tile", DistributionKind::unit_tile)
    .export_values();

  py::enum_<MemoryManagerKind>(m, "MemoryManagerKind")
    .value("invalid", MemoryManagerKind::invalid)
    .value("ga", MemoryManagerKind::ga)
    .value("local", MemoryManagerKind::local)
    .export_values();

  py::enum_<SpinPosition>(m, "SpinPosition")
    .value("ignore", SpinPosition::ignore)
    .value("upper", SpinPosition::upper)
    .value("lower", SpinPosition::lower)
    .export_values();

  py::enum_<IndexPosition>(m, "IndexPosition")
    .value("upper", IndexPosition::upper)
    .value("lower", IndexPosition::lower)
    .value("neither", IndexPosition::neither)
    .export_values();

  py::enum_<SpinType>(m, "SpinType")
    .value("ao_spin", SpinType::ao_spin)
    .value("mo_spin", SpinType::mo_spin)
    .export_values();

  py::enum_<ExecutionHW>(m, "ExecutionHW")
    .value("CPU", ExecutionHW::CPU)
    .value("GPU", ExecutionHW::GPU)
    .value("DEFAULT", ExecutionHW::DEFAULT)
    .export_values();

  py::enum_<ReduceOp>(m, "ReduceOp")
    .value("min", ReduceOp::min)
    .value("max", ReduceOp::max)
    .value("sum", ReduceOp::sum)
    .value("maxloc", ReduceOp::maxloc)
    .value("minloc", ReduceOp::minloc);
}

static void bind_ranges_and_index_spaces(py::module_& m) {
  py::class_<Range>(m, "Range")
    .def(py::init<Index, Index, Index>(), py::arg("lo"), py::arg("hi"), py::arg("step") = 1)
    .def("lo", &Range::lo)
    .def("hi", &Range::hi)
    .def("step", &Range::step)
    .def("contains", &Range::contains)
    .def("overlap_with", &Range::overlap_with)
    .def("is_disjoint_with", &Range::is_disjoint_with);

  m.def("range", py::overload_cast<Index, Index, Index>(&range), py::arg("lo"), py::arg("hi"),
        py::arg("step") = 1);
  m.def("range", py::overload_cast<Index>(&range), py::arg("count"));

  m.def("construct_index_vector",
        [](const Range& r) { return from_index_vector(construct_index_vector(r)); });
  m.def("construct_index_vector", [](const std::vector<Range>& ranges) {
    return from_index_vector(construct_index_vector(ranges));
  });

  py::class_<IndexSpace>(m, "IndexSpace")
    .def(py::init<>())
    .def(py::init<Range, NameToRangeMap, AttributeToRangeMap<Spin>, AttributeToRangeMap<Spatial>>(),
         py::arg("range"), py::arg("named_subspaces") = NameToRangeMap{},
         py::arg("spin")    = AttributeToRangeMap<Spin>{},
         py::arg("spatial") = AttributeToRangeMap<Spatial>{})
    .def(py::init([](py::tuple iterable, NameToRangeMap named_subspaces,
                     AttributeToRangeMap<Spin> spin, AttributeToRangeMap<Spatial> spatial) {
           IndexVector v;
           for(auto item: iterable) v.push_back(item.cast<Index>());
           return IndexSpace(v, named_subspaces, spin, spatial);
         }),
         py::arg("range"), py::arg("named_subspaces") = NameToRangeMap{},
         py::arg("spin")    = AttributeToRangeMap<Spin>{},
         py::arg("spatial") = AttributeToRangeMap<Spatial>{})
    .def(py::init<IndexSpace, Range, NameToRangeMap>(), py::arg("is_"), py::arg("range"),
         py::arg("named_subspaces") = NameToRangeMap{})
    .def(
      py::init([](py::list spaces, std::vector<std::string> names, NameToRangeMap named_subspaces,
                  std::map<std::string, std::vector<std::string>> subspace_references) {
        std::vector<IndexSpace> v;
        v.reserve(py::len(spaces));

        for(auto item: spaces) {
          if(!py::isinstance<IndexSpace>(item)) {
            throw py::type_error("IndexSpace(spaces, ...): first argument must be a list "
                                 "of IndexSpace objects");
          }
          v.push_back(item.cast<const IndexSpace&>());
        }

        return IndexSpace(v, names, named_subspaces, subspace_references);
      }),
      py::arg("spaces"), py::arg("names") = std::vector<std::string>{},
      py::arg("named_subspaces")     = NameToRangeMap{},
      py::arg("subspace_references") = std::map<std::string, std::vector<std::string>>{})
    .def(py::init([](const IndexSpace& parent, py::iterable indices) {
           return IndexSpace(parent, to_index_vector(indices));
         }),
         py::arg("parent"), py::arg("indices"))
    .def(py::init([](py::list dep_spaces_py, py::dict dep_map_py) {
           return IndexSpace(
             py_to_tis_vec_checked(dep_spaces_py, "IndexSpace(dependent_spaces, dependency_map)"),
             py_to_indexspace_dependency_map(dep_map_py));
         }),
         py::arg("dependent_spaces"), py::arg("dependency_map"))
    .def(
      py::init([](py::list dep_spaces_py, const IndexSpace& reference_space, py::dict dep_map_py) {
        return IndexSpace(py_to_tis_vec_checked(dep_spaces_py,
                                                "IndexSpace(dependent_spaces, reference_space, "
                                                "dependency_map)"),
                          reference_space, py_to_indexspace_dependency_map(dep_map_py));
      }),
      py::arg("dependent_spaces"), py::arg("reference_space"), py::arg("dependency_map"))
    .def(
      "index",
      [](IndexSpace& self, Index i, py::object indep_index_obj) {
        if(indep_index_obj.is_none()) return self.index(i, IndexVector{});
        return self.index(i, to_index_vector(indep_index_obj));
      },
      py::arg("i"), py::arg("indep_index") = py::none())
    .def("__getitem__", &IndexSpace::operator[])
    .def(
      "__call__",
      [](const IndexSpace& self, py::object arg) -> IndexSpace {
        if(arg.is_none()) return self(IndexVector{});
        if(py::isinstance<py::str>(arg)) return self(arg.cast<std::string>());
        return self(to_index_vector(arg));
      },
      py::arg("arg") = py::none())
    .def(
      "__iter__", [](const IndexSpace& is) { return py::make_iterator(is.begin(), is.end()); },
      py::keep_alive<0, 1>())
    .def("__len__", &IndexSpace::num_indices)
    .def("num_indices", &IndexSpace::num_indices)
    .def("max_num_indices", &IndexSpace::max_num_indices)
    .def("spin", [](const IndexSpace& self, Index idx) -> Spin { return self.spin(idx); })
    .def("spatial", [](const IndexSpace& self, Index idx) -> Spatial { return self.spatial(idx); })
    .def("spin_ranges", &IndexSpace::spin_ranges, py::return_value_policy::reference_internal)
    .def("spatial_ranges", &IndexSpace::spatial_ranges, py::return_value_policy::reference_internal)
    .def("has_spin", &IndexSpace::has_spin)
    .def("has_spatial", &IndexSpace::has_spatial)
    .def("get_named_ranges", &IndexSpace::get_named_ranges,
         py::return_value_policy::reference_internal)
    .def("root_index_space", &IndexSpace::root_index_space)
    .def("key_tiled_index_spaces", &IndexSpace::key_tiled_index_spaces,
         py::return_value_policy::reference_internal)
    .def("num_key_tiled_index_spaces", &IndexSpace::num_key_tiled_index_spaces)
    .def("map_tiled_index_spaces", &IndexSpace::map_tiled_index_spaces,
         py::return_value_policy::reference_internal)
    .def("map_named_sub_index_spaces", &IndexSpace::map_named_sub_index_spaces,
         py::return_value_policy::reference_internal)
    .def("is_identical", &IndexSpace::is_identical)
    .def("is_less_than", &IndexSpace::is_less_than)
    .def("is_compatible", &IndexSpace::is_compatible)
    .def("is_identical_reference", &IndexSpace::is_identical_reference)
    .def("is_compatible_reference", &IndexSpace::is_compatible_reference)
    .def("is_dependent", &IndexSpace::is_dependent)
    .def("get_spin", &IndexSpace::get_spin)
    .def("get_spatial", &IndexSpace::get_spatial)
    .def("find_pos", &IndexSpace::find_pos)
    .def("hash", &IndexSpace::hash)
    .def(py::self == py::self)
    .def(py::self != py::self)
    .def(py::self < py::self)
    .def(py::self <= py::self)
    .def(py::self > py::self)
    .def(py::self >= py::self);

  py::class_<TiledIndexSpace>(m, "TiledIndexSpace")
    .def(py::init<>())
    .def(py::init<IndexSpace, Tile>(), py::arg("is_"), py::arg("input_tile_size") = 1)
    .def(py::init<IndexSpace, std::vector<Tile>>(), py::arg("is_"), py::arg("input_tile_sizes"))
    .def(py::init<TiledIndexSpace, Range>(), py::arg("t_is"), py::arg("range"))
    .def(py::init([](const TiledIndexSpace& t_is, py::iterable indices_py) {
      return TiledIndexSpace(t_is, to_index_vector(indices_py));
    }))
    .def(py::init(
           [](const TiledIndexSpace& t_is, const TiledIndexSpaceVec& dep_vec, py::dict dep_map_py) {
             std::map<IndexVector, TiledIndexSpace> dep_map;
             for(auto item: dep_map_py) {
               dep_map.emplace(to_index_vector(item.first), item.second.cast<TiledIndexSpace>());
             }
             return TiledIndexSpace(t_is, dep_vec, dep_map);
           }),
         py::arg("t_is"), py::arg("dep_vec"), py::arg("dep_map"))
    .def("ref_indices",
         [](const TiledIndexSpace& self) { return from_index_vector(self.ref_indices()); })
    .def("tiled_dep_map",
         [](const TiledIndexSpace& self) {
           py::dict out;
           for(const auto& kv: self.tiled_dep_map()) {
             out[from_index_vector(kv.first)] = py::cast(kv.second);
           }
           return out;
         })
    .def("index_space", &TiledIndexSpace::index_space)
    .def("label", py::overload_cast<std::string, Label>(&TiledIndexSpace::label, py::const_),
         py::arg("id"), py::arg("lbl") = make_label())
    .def(
      "label",
      [](const TiledIndexSpace& self, py::object id_or_lbl, py::object lbl_obj) {
        if(id_or_lbl.is_none() || py::isinstance<py::str>(id_or_lbl)) {
          const std::string id  = id_or_lbl.is_none() ? std::string{"all"}
                                                      : id_or_lbl.cast<std::string>();
          const Label       lbl = lbl_obj.is_none() ? make_label() : lbl_obj.cast<Label>();
          return self.label(id, lbl);
        }

        if(!lbl_obj.is_none()) throw py::type_error("label(lbl) does not accept a second argument");
        return self.label(id_or_lbl.cast<Label>());
      },
      py::arg("id_or_lbl") = py::none(), py::arg("lbl") = py::none())
    .def(
      "labels",
      [](const TiledIndexSpace& self, std::string id, py::object start_obj,
         size_t number) -> py::object {
        const Label start = start_obj.is_none() ? make_label() : start_obj.cast<Label>();

        // Due to the number of labels being fixed at compile-time, we cannot support a variable
        // number of labels Therefore up to 12 are implemented This can be extended as needed
        switch(number) {
          case 1: return py::cast(self.labels<1>(id, start));
          case 2: return py::cast(self.labels<2>(id, start));
          case 3: return py::cast(self.labels<3>(id, start));
          case 4: return py::cast(self.labels<4>(id, start));
          case 5: return py::cast(self.labels<5>(id, start));
          case 6: return py::cast(self.labels<6>(id, start));
          case 7: return py::cast(self.labels<7>(id, start));
          case 8: return py::cast(self.labels<8>(id, start));
          case 9: return py::cast(self.labels<9>(id, start));
          case 10: return py::cast(self.labels<10>(id, start));
          case 11: return py::cast(self.labels<11>(id, start));
          case 12: return py::cast(self.labels<12>(id, start));
          default: throw py::value_error("These bindings support up to 12 labels");
        }
      },
      py::arg("id") = "all", py::arg("start") = py::none(), py::arg("count") = 1)
    .def(
      "__iter__",
      [](const TiledIndexSpace& tis) { return py::make_iterator(tis.begin(), tis.end()); },
      py::keep_alive<0, 1>())
    .def("__len__", [](const TiledIndexSpace& tis) { return tis.num_tiles(); })
    .def("num_tiles", py::overload_cast<>(&TiledIndexSpace::num_tiles, py::const_))
    .def("block_begin",
         [](const TiledIndexSpace& self, Index i) -> Index { return *self.block_begin(i); })
    .def("block_end",
         [](const TiledIndexSpace& self, Index i) -> Index { return *(self.block_end(i) - 1); })
    .def("is_compatible_with", &TiledIndexSpace::is_compatible_with)
    .def("tile_size", &TiledIndexSpace::tile_size)
    .def("spin", &TiledIndexSpace::spin)
    .def("__eq__", [](const TiledIndexSpace& a, const TiledIndexSpace& b) { return a == b; })
    .def("__call__",
         [](const TiledIndexSpace& self, py::object arg) -> TiledIndexSpace {
           if(py::isinstance<py::str>(arg)) return self(arg.cast<std::string>());
           return self(to_index_vector(arg));
         })
    .def("max_num_indices", &TiledIndexSpace::max_num_indices)
    .def("input_tile_size", &TiledIndexSpace::input_tile_size);

  py::class_<TiledIndexLabel>(m, "TiledIndexLabel")
    .def(py::init<>())
    .def("tiled_index_space", &TiledIndexLabel::tiled_index_space)
    .def("__call__",
         [](const TiledIndexLabel& self, py::args args) {
           if(args.size() == 0) return TiledIndexLabel{self};

           std::vector<TileLabelElement> secondary_labels;
           secondary_labels.reserve(args.size());

           for(auto obj: args) {
             auto lbl = obj.cast<TiledIndexLabel>();
             secondary_labels.emplace_back(lbl.primary_label());
           }

           return TiledIndexLabel{self, secondary_labels};
         })
    .def("__le__", [](const TiledIndexLabel& a, const TiledIndexLabel& b) { return a <= b; })
    .def("__ge__", [](const TiledIndexLabel& a, const TiledIndexLabel& b) { return a >= b; })
    .def("__eq__", [](const TiledIndexLabel& a, const TiledIndexLabel& b) { return a == b; })
    .def("primary_label", &TiledIndexLabel::primary_label)
    .def("secondary_labels", &TiledIndexLabel::secondary_labels);
}

static void bind_loop_nests(py::module_& m) {
  py::class_<IndexLoopBound>(m, "IndexLoopBound")
    .def(py::init<TiledIndexLabel, std::vector<TiledIndexLabel>, std::vector<TiledIndexLabel>>(),
         py::arg("this_label"), py::arg("lb_labels") = std::vector<TiledIndexLabel>{},
         py::arg("ub_labels") = std::vector<TiledIndexLabel>{})
    .def("this_label", &IndexLoopBound::this_label)
    .def("__add__",
         [](const IndexLoopBound& a, const IndexLoopBound& b) {
           if(!(a.this_label() == b.this_label())) {
             throw py::value_error("IndexLoopBound + IndexLoopBound requires matching labels");
           }
           return a + b;
         })
    .def("__le__", [](const IndexLoopBound& a, const IndexLoopBound& b) { return a <= b; })
    .def("__ge__", [](const IndexLoopBound& a, const IndexLoopBound& b) { return a >= b; })
    .def("__eq__", [](const IndexLoopBound& a, const IndexLoopBound& b) { return a == b; });

  py::class_<IndexLoopNest::Iterator>(m, "IndexLoopNestIterator")
    .def("__iter__", [](IndexLoopNest::Iterator& it) -> IndexLoopNest::Iterator& { return it; })
    .def("__next__", [](IndexLoopNest::Iterator& it) {
      if(it.done()) throw py::stop_iteration();
      auto val = *it;
      ++it;
      return from_index_vector(val);
    });

  py::class_<IndexLoopNest>(m, "IndexLoopNest")
    .def(py::init<>())
    .def(py::init<std::vector<IndexLoopBound>>())
    .def(py::init<IndexLoopNest>())
    .def(py::init([](const TiledIndexLabel& lbl) { return IndexLoopNest{IndexLoopBound{lbl}}; }))
    .def(py::init([](const std::vector<TiledIndexLabel>& labels) {
      std::vector<IndexLoopBound> bounds;
      bounds.reserve(labels.size());
      for(const auto& lbl: labels) bounds.emplace_back(lbl);
      return IndexLoopNest{bounds};
    }))
    .def(py::init([](const TiledIndexSpaceVec& tiss) {
      return IndexLoopNest{tiss, {}, {}, {}};
    }))
    .def(
      "__iter__", [](IndexLoopNest& self) { return IndexLoopNest::Iterator{&self}; },
      py::keep_alive<0, 1>());

  py::class_<LabelLoopNest>(m, "LabelLoopNest")
    .def(py::init<>())
    .def(py::init<IndexLabelVec>())
    .def(py::init<LabelLoopNest>())
    .def(
      "__iter__",
      [](const LabelLoopNest& lln) { return py::make_iterator(lln.begin(), lln.end()); },
      py::keep_alive<0, 1>());
}

static void bind_execution_contexts(py::module_& m) {
  py::class_<ProcGroup>(m, "ProcGroup")
    .def(py::init<>())
    .def_static("create_world_coll", &ProcGroup::create_world_coll)
    .def("size", &ProcGroup::size)
    .def("rank", &ProcGroup::rank)
    .def_static("world_rank", []() { return ProcGroup::world_rank(); })
    .def("barrier", [](ProcGroup& pg) { pg.barrier(); });

  py::class_<RuntimeEngine>(m, "RuntimeEngine").def(py::init<>());

  py::class_<ExecutionContext>(m, "ExecutionContext")
    .def(py::init<>())
    .def(py::init([](ProcGroup& pg, DistributionKind default_distribution_kind,
                     MemoryManagerKind memory_manager_kind) {
           return std::make_unique<ExecutionContext>(pg, default_distribution_kind,
                                                     memory_manager_kind);
         }),
         py::arg("pg"), py::arg("default_distribution_kind"), py::arg("memory_manager_kind"),
         py::keep_alive<1, 2>())
    .def("exhw", &ExecutionContext::exhw)
    .def("pg", [](ExecutionContext& ec) -> ProcGroup { return ec.pg(); })
    .def("pg_rank", [](const ExecutionContext& ec) { return ec.pg().rank(); })
    .def("pg_size", [](const ExecutionContext& ec) { return ec.pg().size(); })
    .def("print", &ExecutionContext::print)
    .def("nnodes", &ExecutionContext::nnodes)
    .def("ppn", &ExecutionContext::ppn)
    .def("has_gpu", &ExecutionContext::has_gpu)
    .def("get_profile_header", &ExecutionContext::get_profile_header)
    .def("get_profile_data", [](ExecutionContext& ec) { return ec.get_profile_data().str(); })
    .def("print_mem_info", &ExecutionContext::print_mem_info)
    .def("flush_and_sync", &ExecutionContext::flush_and_sync);
}

static void bind_tensor_info(py::module_& m) {
  py::class_<TensorInfo>(m, "TensorInfo")
    .def(py::init<>())
    .def(py::init<TiledIndexSpaceVec, std::vector<std::string>>())
    .def(py::init<TiledIndexSpaceVec, std::vector<std::string>, Char2TISMap,
                  std::vector<std::string>, NonZeroCheck>(),
         py::arg("tis_vec"), py::arg("allowed_strs"), py::arg("char_to_sub_str"),
         py::arg("disallowed_strs") = std::vector<std::string>{},
         py::arg("non_zero_check")  = NonZeroCheck{[](const IndexVector&) -> bool { return true; }})
    .def(py::init<TiledIndexSpaceVec, NonZeroCheck>())
    .def(py::init<TiledIndexSpaceVec, std::vector<IndexLabelVec>>())
    .def(py::init<TiledIndexSpaceVec, std::vector<TiledIndexSpaceVec>>());
}

static void bind_expression_types(py::module_& m) {
  py::class_<PyOpExpr>(m, "OpExpr").def("__repr__", [](const PyOpExpr& self) {
    return "<pytamm.OpExpr with " + std::to_string(self.ops.size()) + " ops>";
  });

  py::class_<PyTensorExpr>(m, "TensorExpr")
    .def("__repr__",
         [](const PyTensorExpr& self) {
           if(self.kind == PyTensorExpr::Kind::Tensor)
             return std::string("<pytamm.TensorExpr tensor>");
           if(self.kind == PyTensorExpr::Kind::Scale)
             return std::string("<pytamm.TensorExpr scale>");
           return std::string("<pytamm.TensorExpr mul>");
         })
    .def(
      "__mul__",
      [](const PyTensorExpr& lhs, py::object rhs) {
        if(py_is_numeric_scalar(rhs)) return PyTensorExpr::scaled(py_to_expr_scalar(rhs), lhs);
        return PyTensorExpr::mul(lhs, py_to_tensor_expr(rhs));
      },
      py::is_operator())
    .def(
      "__rmul__",
      [](const PyTensorExpr& rhs, py::object lhs) {
        if(!py_is_numeric_scalar(lhs)) throw py::type_error("Only scalar * TensorExpr supported");
        return PyTensorExpr::scaled(py_to_expr_scalar(lhs), rhs);
      },
      py::is_operator());

  m.def("mul", [](py::object a, py::object b) {
    return PyTensorExpr::mul(py_to_tensor_expr(a), py_to_tensor_expr(b));
  });

  py::class_<Op, std::shared_ptr<Op>>(m, "Op").def("canonicalize", &Op::canonicalize);

  py::class_<OpList>(m, "OpList")
    .def(py::init<>())
    .def("__len__", [](const OpList& ops) { return ops.size(); })
    .def(
      "__iter__", [](OpList& ops) { return py::make_iterator(ops.begin(), ops.end()); },
      py::keep_alive<0, 1>());

  py::class_<PyDAG>(m, "DAG")
    .def("ops", [](PyDAG& self) { return pydag_to_oplist(self); })
    .def("__repr__", [](const PyDAG&) { return "<pytamm.DAG>"; });

  m.def(
    "make_dag",
    [](py::function func, py::args args) -> PyDAG {
      return PyDAG{func, py::tuple(args)};
    },
    py::arg("func"));
}

static void bind_symbolic_tools(py::module_& m) {
  py::class_<SymbolTable>(m, "SymbolTable")
    .def(py::init<>())
    .def("__len__", [](const SymbolTable& st) { return st.size(); })
    .def("__getitem__",
         [](const SymbolTable& st, std::size_t key) {
           auto it = st.find(reinterpret_cast<void*>(key));
           if(it == st.end()) throw py::key_error("Key not found in SymbolTable");
           return it->second;
         })
    .def("__repr__", [](const SymbolTable& st) {
      return "<pytamm.SymbolTable size=" + std::to_string(st.size()) + ">";
    });

  m.def(
    "register_symbols",
    [](py::object symbol_table_obj, py::kwargs kwargs) {
      auto& symbol_table = symbol_table_obj.cast<SymbolTable&>();
      py_register_symbols_named(symbol_table, kwargs);
    },
    py::arg("symbol_table"));

  py::class_<OpCostCalculator>(m, "OpCostCalculator")
    .def(py::init<SymbolTable&>(), py::arg("symbol_table"), py::keep_alive<1, 2>())
    .def(
      "print_tensor_execution_report",
      [](OpCostCalculator& self, std::shared_ptr<PTd> tensor, bool use_opmin) {
        if(!tensor) throw py::value_error("tensor must not be None");
        self.print_tensor_execution_report(tensor->raw(), use_opmin);
      },
      py::arg("tensor"), py::arg("use_opmin") = true)
    .def(
      "get_total_op_cost",
      [](OpCostCalculator& self, std::shared_ptr<PTd> tensor, bool use_opmin) {
        if(!tensor) throw py::value_error("tensor must not be None");
        auto res = self.get_total_op_cost(tensor->raw(), use_opmin);
        return py::make_tuple(res.first, res.second);
      },
      py::arg("tensor"), py::arg("use_opmin") = false)
    .def(
      "get_op_cost",
      [](OpCostCalculator& self, const std::shared_ptr<NewOpBase>& op, std::shared_ptr<PLTd> lt,
         bool is_update, bool use_opmin) {
        if(!op) throw py::value_error("op must not be None");
        if(!lt) throw py::value_error("labeled_tensor must not be None");
        return self.get_op_cost(op->clone(), lt->materialize(), is_update, use_opmin);
      },
      py::arg("op"), py::arg("labeled_tensor"), py::arg("is_update") = true,
      py::arg("use_opmin") = false)
    .def(
      "get_op_mem_cost",
      [](OpCostCalculator& self, const std::shared_ptr<NewOpBase>& op, std::shared_ptr<PLTd> lt,
         bool use_opmin) {
        if(!op) throw py::value_error("op must not be None");
        if(!lt) throw py::value_error("labeled_tensor must not be None");
        return self.get_op_mem_cost(op->clone(), lt->materialize(), use_opmin);
      },
      py::arg("op"), py::arg("labeled_tensor"), py::arg("use_opmin") = false)
    .def(
      "print_op_binarized",
      [](OpCostCalculator& self, const NewLTOp& lhs, const std::shared_ptr<NewOpBase>& op) {
        if(!op) throw py::value_error("op must not be None");
        self.print_op_binarized(lhs, op->clone());
      },
      py::arg("lhs"), py::arg("op"))
    .def(
      "print_op_binarized",
      [](OpCostCalculator& self, std::shared_ptr<PLTd> lhs, const std::shared_ptr<NewOpBase>& op) {
        if(!lhs) throw py::value_error("lhs must not be None");
        if(!op) throw py::value_error("op must not be None");
        self.print_op_binarized(make_ltop_value_from_labeled_double(lhs), op->clone());
      },
      py::arg("lhs"), py::arg("op"));

  py::class_<OpExecutor>(m, "OpExecutor")
    .def(py::init<Scheduler&, SymbolTable&>(), py::arg("scheduler"), py::arg("symbol_table"),
         py::keep_alive<1, 2>(), py::keep_alive<1, 3>())
    .def(
      "execute",
      [](OpExecutor& self, std::shared_ptr<PTd> tensor, bool use_opmin, ExecutionHW execute_on,
         bool profile) {
        if(!tensor) throw py::value_error("tensor must not be None");
        self.execute(tensor->raw(), use_opmin, execute_on, profile);
      },
      py::arg("tensor"), py::arg("use_opmin") = false, py::arg("execute_on") = ExecutionHW::CPU,
      py::arg("profile") = false)
    .def("opmin_execute",
         [](OpExecutor& self, std::shared_ptr<PTd> tensor) {
           if(!tensor) throw py::value_error("tensor must not be None");
           self.opmin_execute(tensor->raw());
         })
    .def(
      "pretty_print_binarized",
      [](OpExecutor& self, std::shared_ptr<PTd> tensor, bool use_opmin, size_t start_update) {
        if(!tensor) throw py::value_error("tensor must not be None");
        self.pretty_print_binarized(tensor->raw(), use_opmin, start_update);
      },
      py::arg("tensor"), py::arg("use_opmin") = false, py::arg("start_update") = 0)
    .def(
      "print_op_cost",
      [](OpExecutor& self, std::shared_ptr<PTd> tensor, bool use_opmin) {
        if(!tensor) throw py::value_error("tensor must not be None");
        self.print_op_cost(tensor->raw(), use_opmin);
      },
      py::arg("tensor"), py::arg("use_opmin") = false)
    .def("scheduler", &OpExecutor::scheduler, py::return_value_policy::reference_internal)
    .def("symbol_table", &OpExecutor::symbol_table, py::return_value_policy::reference_internal);

  py::class_<OpMin>(m, "OpMin")
    .def(py::init<SymbolTable&>(), py::arg("symbol_table"), py::keep_alive<1, 2>())
    .def(
      "optimize_all",
      [](OpMin& self, const NewLTOp& lhs, const std::shared_ptr<NewOpBase>& op,
         bool is_assign) -> std::shared_ptr<NewOpBase> {
        if(!op) throw py::value_error("op must not be None");
        auto up = self.optimize_all(lhs, *op, is_assign);
        return std::shared_ptr<NewOpBase>(std::move(up));
      },
      py::arg("lhs"), py::arg("op"), py::arg("is_assign") = true)
    .def(
      "optimize_all",
      [](OpMin& self, std::shared_ptr<PLTd> lhs, const std::shared_ptr<NewOpBase>& op,
         bool is_assign) -> std::shared_ptr<NewOpBase> {
        if(!lhs) throw py::value_error("lhs must not be None");
        if(!op) throw py::value_error("op must not be None");
        auto up = self.optimize_all(make_ltop_value_from_labeled_double(lhs), *op, is_assign);
        return std::shared_ptr<NewOpBase>(std::move(up));
      },
      py::arg("lhs"), py::arg("op"), py::arg("is_assign") = true);
}

// -----------------------------------------------------------------------------
// Module
// -----------------------------------------------------------------------------

PYBIND11_MODULE(pytamm, m) {
  bind_basic_types(m);
  bind_ranges_and_index_spaces(m);
  bind_loop_nests(m);
  bind_execution_contexts(m);
  bind_tensor_info(m);
  bind_expression_types(m);

  auto scheduler_cls =
    py::class_<Scheduler>(m, "Scheduler")
      .def(py::init<ExecutionContext&>(), py::arg("ec"), py::keep_alive<1, 2>())
      .def("execute", static_cast<void (Scheduler::*)(ExecutionHW, bool)>(&Scheduler::execute),
           py::arg("execute_on") = ExecutionHW::CPU, py::arg("profile") = false)
      .def_static(
        "execute_dag",
        [](PyDAG& dag, ExecutionContext& ec, ExecutionHW execute_on, bool profile) {
          py::gil_scoped_acquire gil;

          PyTensorCollector persistent;
          collect_pyobject_tensors(dag.args, persistent);

          dag.materialized = dag.func(*dag.args);

          py::object specs_obj   = dag.materialized;
          py::object tensors_obj = py::none();

          if(py::isinstance<py::tuple>(dag.materialized)) {
            auto tup = dag.materialized.cast<py::tuple>();
            if(py::len(tup) >= 1) specs_obj = tup[0];
            if(py::len(tup) >= 2) tensors_obj = tup[1];
          }

          PyTensorCollector all;
          collect_pyobject_tensors(specs_obj, all);
          collect_pyobject_tensors(tensors_obj, all);

          allocate_collected_tensors(all, ec);

          std::vector<AnyPyTensor> temps;
          for(auto& t: all.tensors) {
            if(!persistent.contains(t)) temps.push_back(t);
          }

          auto cleanup = [&](bool swallow) {
            for(auto it = temps.rbegin(); it != temps.rend(); ++it) {
              try {
                deallocate_tensor_variant(*it);
              } catch(...) {
                if(!swallow) throw;
              }
            }
          };

          try {
            OpList ops = py_specs_to_oplist(specs_obj.cast<py::iterable>());

            Scheduler sch{ec};
            for(const auto& op: ops) sch(*op);

            sch.execute(execute_on, profile);
            cleanup(false);
          } catch(...) {
            cleanup(true);
            throw;
          }
        },
        py::arg("dag"), py::arg("ec"), py::arg("execute_on") = ExecutionHW::CPU,
        py::arg("profile") = false)
      .def(
        "allocate",
        [](Scheduler& s, py::args args) -> Scheduler& {
          return scheduler_alloc_or_dealloc(s, args, true);
        },
        py::return_value_policy::reference_internal)
      .def(
        "deallocate",
        [](Scheduler& s, py::args args) -> Scheduler& {
          return scheduler_alloc_or_dealloc(s, args, false);
        },
        py::return_value_policy::reference_internal)
      .def("ec", &Scheduler::ec, py::return_value_policy::reference_internal);

  // Currently binds tensors of type double and std::complex<double>
  bind_tensor_family<double>(m, scheduler_cls);
  bind_tensor_family<std::complex<double>>(m, scheduler_cls);

  bind_numpy_conversions(m);

  bind_local_tensor_family<double>(m, scheduler_cls);
  bind_local_tensor_family<std::complex<double>>(m, scheduler_cls);

  bind_tamm_utils_for_type<double>(m);
  bind_tamm_utils_for_type<std::complex<double>>(m);

  bind_new_ops(m);

  m.def("invert_tis", &invert_tis);
  m.def("compose_tis", &compose_tis);
  m.def("intersect_tis", &intersect_tis);
  m.def("union_tis", &union_tis);
  m.def("project_tis", &project_tis);

  bind_symbolic_tools(m);

  m.def(
    "print_tensor_all",
    [](std::shared_ptr<PTd> t) {
      if(!t) throw py::value_error("tensor must not be None");
      print_tensor_all(t->raw());
    },
    py::arg("tensor"));

  m.def("tamm_git_info", &tamm_git_info);

  m.def(
    "initialize",
    [](std::vector<std::string> args, bool is_mpi_tm) {
      if(args.empty()) args.push_back("python");

      std::vector<char*> argv;
      argv.reserve(args.size());

      for(auto& s: args) argv.push_back(s.data());

      int argc = static_cast<int>(argv.size());
      initialize(argc, argv.data(), is_mpi_tm);
    },
    py::arg("args") = std::vector<std::string>{}, py::arg("is_mpi_tm") = false);

  m.def("finalize", &finalize, py::arg("tamm_mpi_finalize") = true);
}
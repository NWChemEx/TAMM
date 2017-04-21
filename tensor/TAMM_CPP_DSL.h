#include <string>

namespace tamm_cpp_dsl {

using tamm_il = std::initializer_list<std::string>;

class Tensor {
 public:
  tamm_il indices;
  int ndim;
  Tensor(): ndim(0) {}
  //explicit Tensor(const tamm_il &indices_) : indices(indices_) { ndim = indices.size(); }

  Tensor& operator() (const tamm_il &indices_){ indices = indices_; return *this;}
  //Tensor& operator() () {return *this;}
};

class Op {
 public:
   Tensor a;
   Tensor b;
  bool isMult;
  explicit Op(Tensor &a_) : a(a_) { isMult = false; }
  Op(Tensor &a_, Tensor &b_) : a(a_), b(b_) { isMult = true; }
  
};

class Operation {
 public:
  double alpha;
  const tamm_il tc_ids;
  const tamm_il tb_ids;
  const tamm_il ta_ids;
  bool isMult;
  Operation(double alpha, const Tensor &c, const Tensor &a, const Tensor &b)
      : alpha(alpha), tc_ids(c.indices), tb_ids(b.indices), ta_ids(a.indices) {
    isMult = true;
  }
  Operation(double alpha, const Tensor &c, const Tensor &a)
      : alpha(alpha), tc_ids(c.indices), ta_ids(a.indices) {
    isMult = false;
  }
};

// Operation* operator=(const Tensor &c, Op o) {
//   //const Op &o = op;
//   if (o.isMult) return new Operation(1.0, c, o.a, o.b);
//   return new Operation(1.0, c, o.a);
// }

Operation* operator+=(const Tensor &c, Op o) {
  //const Op &o = op;
  if (o.isMult) return new Operation(1.0, c, o.a, o.b);
  return new Operation(1.0, c, o.a);
}

Op&& operator*(double alpha, Op a) { return std::move(Op(a.a)); }

// Op* operator*(Tensor &a) { return new Op(a); }

Op&& operator*(double d, Tensor &a) { return std::move(Op(a)); }

Op&& operator*( Tensor &a,  Tensor &b) { return std::move(Op(a, b)); }

};  // namespace tamm_cpp_dsl

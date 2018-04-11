
#include "index_space_sketch.h"

using namespace tammy;

template <typename T>
class Tensor {
 public:
  Tensor() = default;
  Tensor(std::initializer_list<TiledIndexSpace> tis) : tis{tis} {}
 private:
   std::vector<TiledIndexSpace> tis;
};

template<typename T>
void ccsd_e(const TiledIndexSpace& MO, 
            Tensor<T>& de,
            const Tensor<T>& t1,
            const Tensor<T>& t2,
            const Tensor<T>& f1,
            const Tensor<T>& v2) {

    const TiledIndexSpace& O = MO("occ");
    const TiledIndexSpace& V = MO("virt");
    Tensor<T> i1{O, V};

    TiledIndexLabel p1, p2, p3, p4, p5;
    TiledIndexLabel h3, h4, h5, h6;

    std::tie(p1, p2, p3, p4, p5) = MO.range_labels<5>("virt");
    std::tie(h3, h4, h5, h6) = MO.range_labels<4>("occ");

    // i1(h6,p5) = f1(h6,p5);
    // i1(h6,p5) +=  0.5  * t1(p3,h4) * v2(h4,h6,p3,p5);
    // de() =  0;
    // de() += t1(p5,h6) * i1(h6,p5);
    // de() +=  0.25  * t2(p1,p2,h3,h4) * v2(h3,h4,p1,p2);
}

template<typename T>
void driver() {
    // Construction of tiled index space MO from skretch
    IndexSpace MO_IS{range(0,200), {{"occ", {range(0,100)}}, 
                                  {"virt", {range(100,200)}}}};
    TiledIndexSpace MO{MO_IS, 10};

    const TiledIndexSpace& O = MO("occ");
    const TiledIndexSpace& V = MO("virt");
    const TiledIndexSpace& N = MO("all");
    Tensor<T> de{};
    Tensor<T> t1{V, O};
    Tensor<T> t2{V, V, O, O};
    Tensor<T> f1{N, N};
    Tensor<T> v2{N, N, N, N};
    ccsd_e(MO, de, t1, t2, f1, v2);
}

int main() {
    driver<double>();
    return 0;
}


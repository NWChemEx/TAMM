#pragma once
#include <memory>

namespace tamm {
namespace new_ops {
template<typename... Ops>
class OpVisitor;

template<typename Op>
class OpVisitor<Op> {
public:
    virtual ~OpVisitor()       = default;
    virtual void visit(Op& op) = 0;
};

template<typename Op, typename... Ops>
class OpVisitor<Op, Ops...> : public OpVisitor<Ops...> {
public:
    virtual ~OpVisitor()       = default;
    virtual void visit(Op& op) = 0;
    using OpVisitor<Ops...>::visit;
};

template<typename Base>
class Cloneable {
public:
    std::unique_ptr<Base> clone() const & {
        return std::unique_ptr<Base>{clone_impl()};
    }
    std::unique_ptr<Base> clone() && {
        return std::unique_ptr<Base>{std::move(*this).clone_impl()};
    }

private:
    virtual Base* clone_impl() const & = 0;
    virtual Base* clone_impl() &&      = 0;
};

template<typename This, typename... Base>
class InheritWithCloneable : public Base... {
public:
    std::unique_ptr<This> clone() const & {
        return std::unique_ptr<This>{static_cast<This*>(clone_impl())};
    }
    std::unique_ptr<This> clone() && {
        return std::unique_ptr<This>{
          static_cast<This*>(std::move(*this).clone_impl())};
    }

private:
    InheritWithCloneable* clone_impl() const & override {
        return new This{*static_cast<const This*>(this)};
    }
    InheritWithCloneable* clone_impl() && override {
        return new This{std::move(*static_cast<const This*>(this))};
    }
};

class MultOp;

class AddOp;

class LTOp;

class EinSumOp;

class ReshapeOp;

class LambdaOp;

class ParForOp;

class VisitorBase : public OpVisitor<AddOp, MultOp, LTOp, EinSumOp, ReshapeOp, LambdaOp, ParForOp> {
public:
    virtual ~VisitorBase() = default;
    friend void swap(VisitorBase& first, VisitorBase& second) noexcept {
        using std::swap;
        // no-op
    }
};

class Visitable {
public:
    virtual void accept(VisitorBase& vb) = 0;
    virtual ~Visitable()                 = default;
};

template<typename T>
class MakeVisitable : public virtual Visitable {
public:
    void accept(VisitorBase& vb) { vb.visit(static_cast<T&>(*this)); }
};

} // namespace new_ops
} // namespace tamm

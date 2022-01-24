// Copyright 2016 Pacific Northwest National Laboratory

#pragma once

#include <memory>
#include <map>
#include <vector>

namespace tamm {

class SymbolInterface {
public:
 virtual void* get_symbol_ptr() const = 0;
};
class Symbol : public SymbolInterface {
 public:
   void* get_symbol_ptr() const override { return ref_ptr_.get(); }

  Symbol() : ref_ptr_{std::make_shared<char>('c')} {}

 private:
  std::shared_ptr<char> ref_ptr_;
};  // class Symbol

}  // namespace tamm

namespace tamm::internal {

inline void register_symbols_with_pos(
    std::map<void*, std::string>& symbol_table,
    const std::vector<std::string>& names, size_t pos,
    const SymbolInterface& symbol) {
  EXPECTS(pos + 1 == names.size());
  symbol_table[symbol.get_symbol_ptr()] = names[pos];
}

template <typename... Symbols>
void register_symbols_with_pos(std::map<void*, std::string>& symbol_table,
                               const std::vector<std::string>& names,
                               size_t pos, const SymbolInterface& symbol,
                               Symbols&&... symbols) {
  EXPECTS(pos >= 0 && pos < names.size());
  symbol_table[symbol.get_symbol_ptr()] = names[pos];
  register_symbols_with_pos(symbol_table, names, pos + 1,
                            std::forward<Symbols>(symbols)...);
}

}  // namespace tamm::internal

namespace tamm {
template <typename... Symbols>
void register_symbols(std::map<void*, std::string>& symbol_table,
                      std::string names,
                      const SymbolInterface& symbol, Symbols&&... symbols) {

  std::stringstream ss(names);
    std::vector<std::string> names_vec;

    while(ss.good()) {
        std::string symbol_name;
        getline(ss, symbol_name, ',');
        if(symbol_name[0] == ' ') {
          symbol_name = symbol_name.substr(1);
        }
        names_vec.push_back(symbol_name);
    }
  internal::register_symbols_with_pos(symbol_table, names_vec, 0, symbol,
                                      std::forward<Symbols>(symbols)...);
}
}  // namespace tamm

#define TAMM_REGISTER_SYMBOLS(map_, ...)                               \
  tamm::register_symbols(map_, #__VA_ARGS__, \
                         __VA_ARGS__)


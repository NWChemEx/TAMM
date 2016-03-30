#ifndef __ctce_index_h__
#define __ctce_index_h__

#include "typesf2c.h"
#include "define.h"

namespace ctce {

  class Index {

    private:
      IndexName name_;  /*< name of this index */
      Integer value_;     /*< value of this index */
      Integer value_r_;   /*< value(restricted) of this index */
      int ext_sym_group_; /*< external symmetry group of this index, from lhs of the expr (tC) */

    public:
      /**
       * Constructor
       */
      Index() {};

      /**
       * Constructor. Assign data to this index.
       * @param[in] name Name of the index
       * @param[in] range Range of the index
       * @param[in] esg External symmetry group of the index
       */
      Index(const IndexName& name, const int& esg) 
        : name_(name),
        value_(0), 
        value_r_(0), 
        ext_sym_group_(esg) {}

      /**
       * Destructor
       */
      ~Index() {};

      /**
       * Get the name of the index
       * @return name as a string
       */
      inline const IndexName& name() const { return name_; }

      /**
       * Get the value of the index
       * @return value as a Integer
       */
      inline const Integer& value() const { return value_; }

      /**
       * Get the value(restricted) of the index
       * @return value_r as an Integer
       */
      inline const Integer& value_r() const { return value_r_; }

      /**
       * Get the external symmetry group id of the index
       * @return ext_sym_group as an int
       */
      inline const int& ext_sym_group() const { return ext_sym_group_; }

      /**
       * Set the value of this index
       * @param[in] val value of this index
       */
      inline void setValue(const Integer& val) { value_=val; }

      /**
       * Set the restricted value of this index
       * @param[in] val_r restricted value of this index
       */
      inline void setValueR(const Integer& val_r) { value_r_=val_r; }

  };

  inline bool compareValue(const Index& lhs, const Index& rhs) { return (lhs.value() < rhs.value()); }
  inline bool compareExtSymGroup(const Index& lhs, const Index& rhs) { return (lhs.ext_sym_group() < rhs.ext_sym_group()); }

} /* namespace ctce */

#endif /* __ctce_index_h */


#ifndef __KOKKOSBATCHED_LOCATE_ABSMAX_INTERNAL_HPP__
#define __KOKKOSBATCHED_LOCATE_ABSMAX_INTERNAL_HPP__


/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "KokkosBatched_Util.hpp"

namespace KokkosBatched {

  ///
  /// Serial Internal Impl
  /// ==================== 
  struct SerialLocateAbsMaxInternal {
    template<typename ValueType>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const int m,
           const ValueType *__restrict__ v, const int vs,
           int &idx) {
      typedef ValueType value_type;
      typedef Kokkos::Details::ArithTraits<value_type> ats;

      // initialize
      typedef Kokkos::pair<typename ats::mag_type,int> pair_type;
      pair_type tmp(ats::abs(v[0]), 0);

      for (int i=0;i<m;++i) {
        const auto v_at_i = ats::abs(v[i*vs]);
        tmp = tmp.first < v_at_i ? tmp : pair_type(v_at_i, i);
      }
      idx = tmp.second;

      return 0;
    }

  };

} // end namespace KokkosBatched


#endif

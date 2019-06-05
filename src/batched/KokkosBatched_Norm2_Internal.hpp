#ifndef __KOKKOSBATCHED_NORM2_INTERNAL_HPP__
#define __KOKKOSBATCHED_NORM2_INTERNAL_HPP__


/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "KokkosBatched_Util.hpp"

namespace KokkosBatched {

  ///
  /// Serial Internal Impl
  /// ==================== 
  struct SerialNorm2Internal {
    template<typename ValueType,
             typename MagnitudeType>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const int m,
           const ValueType *__restrict__ v, const int vs,
           MagnitudeType &norm) {
      typedef ValueType value_type;
      typedef MagnitudeType mag_type;
      typedef Kokkos::Details::ArithTraits<value_type> ats;
      typedef Kokkos::Details::ArithTraits<mag_type> mts;

      norm = mag_type(0);
#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
      for (int i=0;i<m;++i) {
        const auto v_at_i = v[i*vs];
        norm += ats::real(v_at_i*ats::conj(v_at_i));
      }
      norm = mts::sqrt(norm);

      return 0;
    }

  };

} // end namespace KokkosBatched


#endif

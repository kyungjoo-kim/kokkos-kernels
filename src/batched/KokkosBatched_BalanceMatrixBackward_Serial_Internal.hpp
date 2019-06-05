#ifndef __KOKKOSBATCHED_BALANCEMATRIX_BACKWARD_SERIAL_INTERNAL_HPP__
#define __KOKKOSBATCHED_BALANCEMATRIX_BACKWARD_SERIAL_INTERNAL_HPP__


/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "KokkosBatched_Util.hpp"
#include "KokkosBatched_SwapVector_Internal.hpp"

namespace KokkosBatched {

  ///
  /// Serial Internal Impl
  /// ==================== 
  ///
  /// permute a given square matrix A as block upper triangular matrix
  /// where A(ibeg:iend,ibeg:iend) becomes a general matrix.
  /// additionally, it also scale the matrix
  ///
  template<typename SideType>
  struct SerialBalanceMatrixBackwardScaleInternal {
    template<typename ValueType>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const int m, const int n,
           const int abeg, const int aend,
           const ValueType * scale, const int ss,
           ValueType * U, const int us0, const int us1) {
      for (int i=abeg;i<aend;++i) {
        const value_type sca = scale[i];
        const value_type val = std::is_same<SideType,Side::Left> ? one/sca : sca;
        SerialScal::invoke(n, val, U+i*as0, as1);
      }
    }
  };      

  struct SerialBalanceMatrixBackwardPermuteInternal {
    template<typename ValueType>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const int m, const int n,
           const int abeg, const int aend,
           const ValueType * scale, const int ss,
           ValueType * U, const int us0, const int us1) {
      for (int ii=0;ii<m;++ii) {
        if (ii >= abeg && ii < aend) {
          // do nothing
        } else {
          const int i = ii < abeg ? abeg - ii : ii;
          const int k = static_cast<int>(scale[i]);
          SerialSwapVectorInternal::invoke(n, U+i*us0, us1, U+k*us0, us1); 
        }
      }
    }
  };      

  struct SerialBalanceMatrixBackwardRightInternal {
    template<typename ValueType>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const int m, const int n,
           const int abeg, const int aend,
           const ValueType * scale, const int ss,
           ValueType * U, const int us0, const int us1) {
      SerialBalanceMatrixBackwardScaleInternal<Side::Right>
        ::invoke(m, n, 
                 abeg, aend,
                 scale, ss,
                 U, us0, us1);

      SerialBalanceMatrixBackwardPermuteInternal
        ::invoke(m, n, abeg, aend,
                 scale, ss, 
                 U, us0, us1);

      return 0;
    }
  };      

  struct SerialBalanceMatrixBackwardLeftInternal {
    template<typename ValueType>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const int m, const int n,
           const int abeg, const int aend,
           const ValueType * scale, const int ss,
           ValueType * U, const int us0, const int us1) {
      SerialBalanceMatrixBackwardScaleInternal<Side::Left>
        ::invoke(m, n, 
                 abeg, aend,
                 scale, ss,
                 U, us0, us1);

      SerialBalanceMatrixBackwardPermuteInternal
        ::invoke(m, n, abeg, aend,
                 scale, ss, 
                 U, us0, us1);

      return 0;
    }
  };      

} // end namespace KokkosBatched


#endif

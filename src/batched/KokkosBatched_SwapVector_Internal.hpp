#ifndef __KOKKOSBATCHED_SWAPVECTOR_INTERNAL_HPP__
#define __KOKKOSBATCHED_SWAPVECTOR_INTERNAL_HPP__


/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "KokkosBatched_Util.hpp"

namespace KokkosBatched {

  ///
  /// Serial Internal Impl
  /// ==================== 
  struct SerialSwapVectorInternal {
    template<typename ValueType>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const int m, 
           /* */ ValueType *__restrict__ A, const int as,
           /* */ ValueType *__restrict__ B, const int bs) {

#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
      for (int i=0;i<m;++i) {
        const int ias = i*as, ibs = i*bs;
        const ValueType tmp = A[ias];
        A[ias] = B[ibs];
        B[ibs] = tmp;
      }
      return 0;
    }
  };

  ///
  /// Vector Internal Impl
  /// ==================== 
  struct VectorSwapVectorInternal {
    template<typename ValueType>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const int m, 
           /* */ ValueType *__restrict__ A, const int as,
           /* */ ValueType *__restrict__ B, const int bs) {

      Kokkos::parallel_for
        (Kokkos::ThreadVectorRange(member,0,m),
         [&](const int &i) {
          const int ias = i*as, ibs = i*bs;
          const ValueType tmp = A[ias];
          A[ias] = B[ibs];
          B[ibs] = tmp;
        });
      return 0;
    }
  };

  ///
  /// Team Internal Impl
  /// ================== 
  struct TeamSwapVectorInternal {
    template<typename ValueType>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const int m, 
           /* */ ValueType *__restrict__ A, const int as,
           /* */ ValueType *__restrict__ B, const int bs) {
      Kokkos::parallel_for
        (Kokkos::TeamThreadRange(member,0,m),
         [&](const int &i) {
          const int ias = i*as, ibs = i*bs;
          const ValueType tmp = A[ias];
          A[ias] = B[ibs];
          B[ibs] = tmp;
        });
      return 0;
    }
  };

  // ///
  // /// TeamVector Internal Impl
  // /// ======================== 
  // struct TeamSwapVectorInternal {
  //   template<typename ValueType>
  //   KOKKOS_INLINE_FUNCTION
  //   static int
  //   invoke(const int m, 
  //          /* */ ValueType *__restrict__ A, const int as,
  //          /* */ ValueType *__restrict__ B, const int bs) {
  //     Kokkos::parallel_for
  //       (Kokkos::TeamVectorThreadRange(member,0,m),
  //        [&](const int &i) {
  //         const int ias = i*as, ibs = i*bs;
  //         const ValueType tmp = A[ias];
  //         A[ias] = B[ibs];
  //         B[ibs] = tmp;
  //       });
  //     return 0;
  //   }
  // };

} // end namespace KokkosBatched


#endif

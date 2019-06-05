#ifndef __KOKKOSBATCHED_BALANCEMATRIX_SERIAL_INTERNAL_HPP__
#define __KOKKOSBATCHED_BALANCEMATRIX_SERIAL_INTERNAL_HPP__


/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "KokkosBatched_Util.hpp"
#include "KokkosBatched_SwapVector_Internal.hpp"
#include "KokkosBatched_Norm2_Internal.hpp"
#include "KokkosBatched_LocateAbsMax_Internal.hpp"
#include "KokkosBatched_Scale_Internal.hpp"

namespace KokkosBatched {

  ///
  /// Serial Internal Impl
  /// ==================== 
  ///
  /// permute a given square matrix A as block upper triangular matrix
  /// where A(ibeg:iend,ibeg:iend) becomes a general matrix.
  /// additionally, it also scale the matrix
  ///
  struct SerialBalanceNoneMatrixInternal {
    template<typename ValueType>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const int m, 
           ValueType * A, const int as0, const int as1,
           int &abeg, int &aend,
           ValueType * scale, const int ss) {
      abeg = 0; aend = m;
      const ValueType one(1);
      for (int i=0;i<m;++i) 
        scale[ss*i] = one;
      return 0;
    }
  };

  struct SerialBalancePermuteMatrixInternal {
    template<typename ValueType>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const int m, 
           ValueType * A, const int as0, const int as1,
           int &abeg, int &aend,
           ValueType * scale, const int ss) {
      typedef ValueType value_type;
      typedef Kokkos::Details::ArithTraits<value_type> ats;
      typedef typename ats::mag_type magnitude_type;

      const value_type zero(0);

      // swap vectors
      auto permute = [&](const int mm, const int jj, const int ll, const int kk) {
        scale[mm*ss] = jj;
        if (mm == jj) { 
          // do nothing
        } else {
          SerialSwapVectorInternal::invoke(ll,   A+jj*as1,        as0, A+mm*as1,        as0); // colum swap
          SerialSwapVectorInternal::invoke(m-kk, A+jj*as0+kk*as1, as1, A+mm*as0+kk*as1, as1); // row swap          
        }
      };
      
      // search for rows isolating and eigenvalue and push them down
      auto search_rows = [&](const int k, const int l) -> bool {
        for (int jj=0;jj<l;++jj) {
          bool is_permute = true;
          const int j = l-jj-1;
          for (int i=0;i<l;++i) {
            if (i == j) continue;
            if (A[j*as0+i*as1] != zero) {
              is_permute = false;
              break;
            }
          }
          if (is_permute) {
            permute(l-1, j, l, k);
            return true;
          }
        }
        return false;
      };

      // search for columns isolating and eigenvalue and push them left
      auto search_cols = [&](const int k, const int l) -> bool {
        for (int j=k;j<l;++j) {
          bool is_permute = true;
          for (int i=k;i<l;++i) {
            if (i == j) continue;
            if (A[i*as0+j*as1] != zero) { 
              is_permute = false; 
              break; 
            }
          }
          if (is_permute) {
            permute(k, j, l, k); 
            return true;
          }
        }
        return false;
      };
      
      /// permute to isolate eigenvalues if possible
      {
        int l=m, k=0;
        for (;l>0 && search_rows(k,l);--l);
        for (;k<l && search_cols(k,l);++k);
        abeg = k;
        aend = l;
      }
      return 0;
    }
  };

  struct SerialBalanceScaleMatrixInternal {
    template<typename ValueType>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const int m, const int abeg, const int aend,
           ValueType * A, const int as0, const int as1,
           ValueType * scale, const int ss) {
      typedef ValueType value_type;
      typedef Kokkos::Details::ArithTraits<value_type> ats;
      typedef typename ats::mag_type magnitude_type;

      const value_type one(1), zero(0);
      const magnitude_type factor(0.95), sclfac(2.0);
      
      const magnitude_type sfmin1 = ats::sfmin()/static_cast<magnitude_type>(ats::base());
      const magnitude_type sfmax1 = one/sfmin1;

      const magnitude_type sfmin2 = sfmin1*sclfac;
      const magnitude_type sfmax2 = one/sfmin2;

      {
        const int k = abeg, l = aend, lk = l - k;
        for (int i=k;i<l;++i) 
          scale[i*ss] = one;
        
        bool converged(false);
        do {
          converged = true;
          for (int i=k;i<l;++i) {
            magnitude_type c, r;
            SerialNorm2Internal::invoke(lk, A+k*as0+i*as1, as0, c);
            SerialNorm2Internal::invoke(lk, A+i*as0+k*as1, as1, r);
            
            int idx_c, idx_r;
            SerialLocateAbsMaxInternal::invoke(l,   A+      i*as1, as0, idx_c);
            SerialLocateAbsMaxInternal::invoke(m-k, A+i*as0+k*as1, as1, idx_r);
            
            magnitude_type ca = ats::abs(A[idx_c*as0+i*as1]);
            magnitude_type ra = ats::abs(A[i*as0+(idx_r+k)*as1]);

            // skip for c and r are zeros
            if (c == zero || r == zero) continue;
            
            // find out balance factor
            magnitude_type g(r/sclfac), f(one), s(c+r);
            while (true) {
              const magnitude_type max_f_c_ca = f > c ? (f > ca ? f : ca) : (c > ca ? c : ca);
              const magnitude_type min_r_g_ra = r < g ? (r < ra ? r : ra) : (g < ra ? g : ra);
              if (c >= g || max_f_c_ca >= sfmax2 || min_r_g_ra <= sfmin2) break;
              f  *= sclfac;
              c  *= sclfac;
              ca *= sclfac;
              r  /= sclfac;
              g  /= sclfac;
              ra /= sclfac;
            }            
            g = c/sclfac;
            while (true) {
              const magnitude_type max_r_ra = r > ra ? r : ra;
              const magnitude_type min_f_c = f < c ? f : c;
              const magnitude_type min_g_ca = g < ca ? g :ca;
              const magnitude_type min_f_c_g_ca = min_f_c < min_g_ca ? min_f_c : min_g_ca;
              if (g <= r || max_r_ra >= sfmax2 || min_f_c_g_ca <= sfmin2) break;
              f  /= sclfac;
              c  /= sclfac;
              g  /= sclfac;
              ca /= sclfac;
              r  *= sclfac;
              ra *= sclfac;
            }
            
            /// now balance
            if ((c+r) >= factor*s) continue;
            if (f < one && scale[i*ss] < one) 
              if (f*scale[i*ss] <= sfmin1) 
                continue;
            if (f > one && scale[i*ss] > one)
              if (scale[i*ss] >= sfmax1/f)
                continue;
            
            // not converged          
            converged = false;

            g = one/f;
            scale[i*ss] *= f;
            
            SerialScaleInternal::invoke(m-k, g, A+i*as0+k*as1, as1);
            SerialScaleInternal::invoke(l,   f, A+      i*as1, as0);
          }
        } while (!converged);
        
      }
      return 0;
    }
  };

  struct SerialBalanceMatrixInternal {
    template<typename ValueType>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const int m, 
           ValueType * A, const int as0, const int as1,
           int &abeg, int &aend,
           ValueType * scale, const int ss) {
      SerialBalancePermuteMatrixInternal::invoke(m, A, as0, as1, abeg, aend, scale, ss);
      SerialBalanceScaleMatrixInternal::invoke(m, abeg, aend, A, as0, as1, scale, ss);
      return 0;
    }
  };
  
} // end namespace KokkosBatched


#endif

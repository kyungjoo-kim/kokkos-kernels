#!/bin/bash
KOKKOSKERNELS_PATH=$1
cd ${KOKKOSKERNELS_PATH}/src/impl
mkdir generated_specializations_hpp
mkdir generated_specializations_cpp

#abs
${KOKKOSKERNELS_PATH}/scripts/generate_specialization_function2.bash abs KokkosBlas1_abs KokkosBlas1_abs_spec.hpp KokkosBlas ${KOKKOSKERNELS_PATH}
${KOKKOSKERNELS_PATH}/scripts/generate_specialization_function2.bash abs KokkosBlas1_abs_mv KokkosBlas1_abs_spec.hpp KokkosBlas ${KOKKOSKERNELS_PATH}

#axpby
${KOKKOSKERNELS_PATH}/scripts/generate_specialization_function2.bash axpby KokkosBlas1_axpby KokkosBlas1_axpby_spec.hpp KokkosBlas ${KOKKOSKERNELS_PATH}
${KOKKOSKERNELS_PATH}/scripts/generate_specialization_function2.bash axpby KokkosBlas1_axpby_mv KokkosBlas1_axpby_spec.hpp KokkosBlas ${KOKKOSKERNELS_PATH}

#dot
${KOKKOSKERNELS_PATH}/scripts/generate_specialization_function2.bash dot KokkosBlas1_dot KokkosBlas1_dot_spec.hpp KokkosBlas ${KOKKOSKERNELS_PATH}
${KOKKOSKERNELS_PATH}/scripts/generate_specialization_function2.bash dot KokkosBlas1_dot_mv KokkosBlas1_dot_spec.hpp KokkosBlas ${KOKKOSKERNELS_PATH}

#nrm1
${KOKKOSKERNELS_PATH}/scripts/generate_specialization_function2.bash nrm1 KokkosBlas1_nrm1 KokkosBlas1_nrm1_spec.hpp KokkosBlas ${KOKKOSKERNELS_PATH}
${KOKKOSKERNELS_PATH}/scripts/generate_specialization_function2.bash nrm1 KokkosBlas1_nrm1_mv KokkosBlas1_nrm1_spec.hpp KokkosBlas ${KOKKOSKERNELS_PATH}

#nrm2
${KOKKOSKERNELS_PATH}/scripts/generate_specialization_function2.bash nrm2 KokkosBlas1_nrm2 KokkosBlas1_nrm2_spec.hpp KokkosBlas ${KOKKOSKERNELS_PATH}
${KOKKOSKERNELS_PATH}/scripts/generate_specialization_function2.bash nrm2 KokkosBlas1_nrm2_mv KokkosBlas1_nrm2_spec.hpp KokkosBlas ${KOKKOSKERNELS_PATH}

#nrm2w
${KOKKOSKERNELS_PATH}/scripts/generate_specialization_function2.bash nrm2w KokkosBlas1_nrm2w KokkosBlas1_nrm2w_spec.hpp KokkosBlas ${KOKKOSKERNELS_PATH}
${KOKKOSKERNELS_PATH}/scripts/generate_specialization_function2.bash nrm2w KokkosBlas1_nrm2w_mv KokkosBlas1_nrm2w_spec.hpp KokkosBlas ${KOKKOSKERNELS_PATH}

#nrminf
${KOKKOSKERNELS_PATH}/scripts/generate_specialization_function2.bash nrminf KokkosBlas1_nrminf KokkosBlas1_nrminf_spec.hpp KokkosBlas ${KOKKOSKERNELS_PATH}
${KOKKOSKERNELS_PATH}/scripts/generate_specialization_function2.bash nrminf KokkosBlas1_nrminf_mv KokkosBlas1_nrminf_spec.hpp KokkosBlas ${KOKKOSKERNELS_PATH}

#reciprocal
${KOKKOSKERNELS_PATH}/scripts/generate_specialization_function2.bash reciprocal KokkosBlas1_reciprocal KokkosBlas1_reciprocal_spec.hpp KokkosBlas ${KOKKOSKERNELS_PATH}
${KOKKOSKERNELS_PATH}/scripts/generate_specialization_function2.bash reciprocal KokkosBlas1_reciprocal_mv KokkosBlas1_reciprocal_spec.hpp KokkosBlas ${KOKKOSKERNELS_PATH}

#scal
${KOKKOSKERNELS_PATH}/scripts/generate_specialization_function2.bash scal KokkosBlas1_scal KokkosBlas1_scal_spec.hpp KokkosBlas ${KOKKOSKERNELS_PATH}
${KOKKOSKERNELS_PATH}/scripts/generate_specialization_function2.bash scal KokkosBlas1_scal_mv KokkosBlas1_scal_spec.hpp KokkosBlas ${KOKKOSKERNELS_PATH}

#sum
${KOKKOSKERNELS_PATH}/scripts/generate_specialization_function2.bash sum KokkosBlas1_sum KokkosBlas1_sum_spec.hpp KokkosBlas ${KOKKOSKERNELS_PATH}
${KOKKOSKERNELS_PATH}/scripts/generate_specialization_function2.bash sum KokkosBlas1_sum_mv KokkosBlas1_sum_spec.hpp KokkosBlas ${KOKKOSKERNELS_PATH}

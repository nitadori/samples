/*!
 * @author    PEZY Computing, K.K.
 * @date      2019
 * @copyright BSD-3-Clause
 * @brief     Wrapper header file for atomic functions
 * @details   These defines will be included by pzsdk near future.
 *            For the convenient use of the OpenCL atomic functions as below,
 *            you can include this file.
 */
// clang-format off
#define atomic_add     pz_atomic_add
#define atomic_sub     pz_atomic_sub
#define atomic_xchg    pz_atomic_xchg
#define atomic_inc     pz_atomic_inc
#define atomic_dec     pz_atomic_dec
#define atomic_cmpxchg pz_atomic_cmpxchg
#define atomic_min     pz_atomic_min
#define atomic_max     pz_atomic_max
#define atomic_and     pz_atomic_and
#define atomic_or      pz_atomic_or
#define atomic_xor     pz_atomic_xor
#define atomic_load    pz_atomic_load
#define atomic_flush   pz_atomic_flush

#include <stdint.h>

typedef int64_t* BitVector ;

/** Initialize preparatory work before running sparsity detection code.
 * Preparatory work is a sequence of initializations of the bitvectors,
 * i.e. a sequence of adSparsity_initScalar's and adSparsity_initArray's.
 * that will initialize each bitvector to an "identity" structural dependency.
 * Argument nbints must be the number of int64's in the bitvectors.
 * Arguments offsetIndeps and nbIndeps are used to restrict computation
 * to columns between rank offsetIndeps and offsetIndeps+nbIndeps-1, where
 * these "ranks" are defined by the order in which these variables are initialized. */
void adSparsity_init(int nbints, int offsetIndeps, int nbIndeps) ;

/** Initialize to "identity" the bitvector that holds the structural dependencies of an input scalar */
void adSparsity_initScalar(char* varname, BitVector indepd) ;

/** Initialize to "identity" the bitvectors that hold the structural dependencies of an input array */
void adSparsity_initArray(char* varname, BitVector indepd, int length) ;

/** Preparation before printing the reults of sparsity analysis */
void adSparsity_startConclude() ;

/** Display the bitvector that holds the structural dependencies of an output scalar */
void adSparsity_concludeScalar(char* varname, BitVector depd) ;

/** Display the bitvectors that hold the structural dependencies of an output array */
void adSparsity_concludeArray(char* varname, BitVector depd, int length) ;

/** Conclusion after printing the reults of sparsity analysis */
void adSparsity_conclude() ;


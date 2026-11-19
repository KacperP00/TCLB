#include "adSparsity.h"

// TEMPORARY:
#include <stdio.h>

static int adsp_nbInts = 1 ;
static int adsp_offsetIndeps = 0 ;
static int adsp_nbIndeps = -1 ;

static int indepIndex = 0 ;
static int depIndex = 0 ;

// Utilities:

void adsp_zero(BitVector bv) {
  int i ;
  for (i=0 ; i<adsp_nbInts ; ++i) bv[i]=0 ;
}

void adsp_setBit(BitVector bv, int index) {
  int intIndex = index/64 ;
  int bitIndex = index%64 ;
  bv[intIndex] |= ((int64_t)1)<<bitIndex ;
  //  printf("AT INDEX %i=>%i/%i --> %016llX \n", index, intIndex, bitIndex, bv[intIndex]) ;
}

int adsp_getBit(BitVector bv, int index) {
  int intIndex = index/64 ;
  int bitIndex = index%64 ;
  return (bv[intIndex]>>bitIndex)%2 ;
}

void adsp_dump(BitVector bv) {
  int index = 0 ;
  int nbBits = indepIndex-adsp_offsetIndeps ;
  if (adsp_nbIndeps!=-1 && nbBits>adsp_nbIndeps) {
    nbBits = adsp_nbIndeps ;
  }
  for (index=0 ; index<nbBits ; ++index) {
    if (adsp_getBit(bv, index)) {
      printf("X") ;
    } else {
      printf(".") ;
    }
  }
}

// Exported functions:

void adSparsity_init(int nbints, int offsetIndeps, int nbIndeps) {
  adsp_nbInts = nbints ;
  indepIndex = 0 ;
  adsp_offsetIndeps = offsetIndeps ;
  adsp_nbIndeps = nbIndeps ;
}

void adSparsity_initScalar(char* varname, BitVector indepd) {
  adsp_zero(indepd) ;
  ++indepIndex ;
  if (adsp_offsetIndeps<indepIndex
      && (adsp_nbIndeps==-1
          || (indepIndex-1-adsp_offsetIndeps)<adsp_nbIndeps)) {
    adsp_setBit(indepd, indepIndex-1-adsp_offsetIndeps) ;
printf(" [%03i] ", indepIndex-1-adsp_offsetIndeps) ;
adsp_dump(indepd) ;
printf(" column %s\n", varname) ;
  }
}

void adSparsity_initArray(char* varname, BitVector indepd, int length) {
  BitVector bv = indepd ;
  int i ;
  for (i=0 ; i<length ; ++i) {
    adsp_zero(bv) ;
    ++indepIndex ;
    if (adsp_offsetIndeps<indepIndex
      && (adsp_nbIndeps==-1
          || (indepIndex-1-adsp_offsetIndeps)<adsp_nbIndeps)) {
      adsp_setBit(bv, indepIndex-1-adsp_offsetIndeps) ;
printf(" [%03i] ", indepIndex-1-adsp_offsetIndeps) ;
adsp_dump(bv) ;
printf(" column %s[%i]\n", varname, i) ;
    }
    bv += adsp_nbInts ;
  }
}

void adSparsity_startConclude() {
  int requiredBits = indepIndex-adsp_offsetIndeps ;
  if (adsp_nbIndeps!=-1 && requiredBits>adsp_nbIndeps) {
    requiredBits = adsp_nbIndeps ;
  }
  printf("%i tracked columns require %i int64's\n", requiredBits, (requiredBits/64)+1) ;
  printf("User-given nbInts:%i are%s enough\n", adsp_nbInts, (adsp_nbInts>(requiredBits/64) ? "" : " NOT")) ;
  depIndex = 0 ;
}

void adSparsity_concludeScalar(char* varname, BitVector depd) {
  printf(" [%03i] ", depIndex) ;
  adsp_dump(depd) ;
  printf(" row %s\n", varname) ;
  ++depIndex ;
}

void adSparsity_concludeArray(char* varname, BitVector depd, int length) {
  BitVector bv = depd ;
  int i ;
  for (i=0 ; i<length ; ++i) {
    printf(" [%03i] ", depIndex) ;
    adsp_dump(bv) ;
    printf(" row %s[%i]\n", varname, i) ;
    ++depIndex ;
    bv += adsp_nbInts ;
  }
}

void adSparsity_conclude() {
}

/****************** INTERFACE CALLED FROM FORTRAN *******************/

void adsparsity_init_(int *nbints, int *offsetIndeps, int *nbIndeps) {
  adSparsity_init(*nbints, *offsetIndeps, *nbIndeps) ;
}

void adsparsity_initscalar_(char* varname, BitVector indepd) {
  adSparsity_initScalar(varname, indepd) ;
}

void adsparsity_initarray_(char* varname, BitVector indepd, int *length) {
  adSparsity_initArray(varname, indepd, *length) ;
}

void adsparsity_startconclude_() {
  adSparsity_startConclude() ;
}

void adsparsity_concludescalar_(char* varname, BitVector depd) {
  adSparsity_concludeScalar(varname, depd) ;
}

void adsparsity_concludearray_(char* varname, BitVector depd, int *length) {
  adSparsity_concludeArray(varname, depd, *length) ;
}

void adsparsity_conclude_() {
  adSparsity_conclude() ;
}

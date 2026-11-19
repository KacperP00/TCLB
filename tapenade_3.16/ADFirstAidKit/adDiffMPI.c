#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <mpi.h>

extern void pushInteger4(int val) ;
extern void popInteger4(int* val) ;

//############## UTILITIES  ################
// (see ADFirstAidKit/ampiSupport.c, ...)

struct AMPI_ShadowComm_list {
  struct AMPI_ShadowComm_list *next_p;
  MPI_Comm comm ;
  MPI_Comm shadowComm ;
} ;

/**
 * the enumeration for communication pairings as explained in \ref pairings
 */
enum AMPI_PairedWith_E { 
  AMPI_TO_RECV,
  AMPI_FROM_SEND,
  AMPI_TO_IRECV_WAIT,
  AMPI_TO_IRECV_WAITALL,
  AMPI_FROM_ISEND_WAIT,
  AMPI_FROM_ISEND_WAITALL,
  AMPI_FROM_BSEND,
  AMPI_FROM_RSEND,
};

/**
 * does the request originate with a send or a receive 
 */
enum AMPI_Request_origin_E { 
  AMPI_SEND_ORIGIN,
  AMPI_RECV_ORIGIN
};

struct AMPI_Request_stack {
  struct AMPI_Request_stack *next_p;
  void *buf ;
  void *adjointBuf ;
  int count ;
  MPI_Datatype datatype ;
  int endPoint ;
  int tag ;
  enum AMPI_PairedWith_E pairedWith;
  MPI_Comm comm;
  int isActive;
  enum AMPI_Request_origin_E origin;
} ;

/**
 * user-defined type data
 * only one instance of derivedTypeData exists at once
 * get pointer from getDTypeData, add new stuff with addDTypeData
 */
typedef struct {
  int size;
  int preAlloc;
  int* num_actives;
  /* displacements of first/last active blocks */
  MPI_Aint* first_active_blocks;
  MPI_Aint* last_active_blocks;
  /* need to know last active block length to find last active element */
  int* last_active_block_lengths;
  MPI_Datatype* derived_types;
  int* counts;
  int** arrays_of_blocklengths;
  MPI_Aint** arrays_of_displacements;
  MPI_Datatype** arrays_of_types;
  MPI_Aint* lbs;
  MPI_Aint* extents;
  /* corresponding typemaps packed for sending */
  MPI_Datatype* packed_types;
  int** arrays_of_p_blocklengths;
  MPI_Aint** arrays_of_p_displacements;
  MPI_Datatype** arrays_of_p_types;
  MPI_Aint* p_extents;
} derivedTypeData;

/**
 * user-defined reduction op data
 * only one instance of userDefinedOpData exists at once
 * get pointer from getUOpData, add new stuff with addUOpData
 */
typedef struct {
  int size;
  int preAlloc;
  MPI_Op* ops;
  MPI_User_function** functions;
  int* commutes;
} userDefinedOpData;






struct AMPI_ShadowComm_list *adDiffMPI_shadowComm_list = NULL ;

//TODO: initialisation a faire... ? :
/* #ifdef AMPI_FORTRANCOMPATIBLE */
/* MPI_Datatype AMPI_ADOUBLE_PRECISION; */
/* MPI_Datatype AMPI_AREAL; */
/* MPI_Fint adouble, areal ; */
/* adtool_ampi_fortransetuptypes_(&adouble, &areal); */
/* AMPI_ADOUBLE_PRECISION=MPI_Type_f2c(adouble); */
/* AMPI_AREAL=MPI_Type_f2c(areal); */
/* #endif */

derivedTypeData* getDTypeData() {
  static derivedTypeData* dat = NULL;
  if (dat==NULL) {
    derivedTypeData* newdat = (derivedTypeData*)malloc(sizeof(derivedTypeData));
    newdat->size = 0;
    newdat->preAlloc = 0;
    newdat->num_actives = NULL;
    newdat->first_active_blocks = NULL;
    newdat->last_active_blocks = NULL;
    newdat->last_active_block_lengths = NULL;
    newdat->derived_types = NULL;
    newdat->counts = NULL;
    newdat->arrays_of_blocklengths = NULL;
    newdat->arrays_of_displacements = NULL;
    newdat->arrays_of_types = NULL;
    newdat->lbs = NULL;
    newdat->extents = NULL;
    newdat->packed_types = NULL;
    newdat->arrays_of_p_blocklengths = NULL;
    newdat->arrays_of_p_displacements = NULL;
    newdat->arrays_of_p_types = NULL;
    newdat->p_extents = NULL;
    dat = newdat;
  }
  return dat;
}

int derivedTypeIdx(MPI_Datatype datatype) {
  int i;
  derivedTypeData* dtdata = getDTypeData();
  for (i=0;i<dtdata->size;i++) {
    if (dtdata->derived_types[i]==datatype) return i;
  }
  return -1;
}

int isDerivedType(int dt_idx) {
  return dt_idx!=-1;
}

/**
 */
userDefinedOpData* getUOpData() {
  static userDefinedOpData* dat = NULL;
  if (dat==NULL) {
    userDefinedOpData* newdat = (userDefinedOpData*)malloc(sizeof(userDefinedOpData));
    newdat->size = 0;
    newdat->preAlloc = 0;
    newdat->ops = NULL;
    newdat->functions = NULL;
    newdat->commutes = NULL;
    dat = newdat;
  }
  return dat;
}

/**
 * @param dat
 * @param op a user-defined operation
 * @param function
 * @param commute
 * takes user-defined op  and adds a new entry;
 * doubles data struct size every
 * time there's overflow
 */
void addUOpData(userDefinedOpData* dat,
                MPI_Op* op,
                MPI_User_function* function,
                int commute) {
  assert(dat);
  if (dat->preAlloc == dat->size) {
    dat->preAlloc += 16;
    dat->ops = realloc(dat->ops,(dat->preAlloc)*sizeof(MPI_Op));
    dat->functions = realloc(dat->functions,(dat->preAlloc)*sizeof(MPI_User_function*));
    dat->commutes = realloc(dat->commutes,(dat->preAlloc)*sizeof(int));
  }
  dat->ops[dat->size] = *op;
  dat->functions[dat->size] = function;
  dat->commutes[dat->size] = commute;
  dat->size += 1;
}


/**
 * @param op
 * @return
 */
int userDefinedOpIdx(MPI_Op op) {
  int i;
  userDefinedOpData* uopdata = getUOpData();
  for (i=0;i<uopdata->size;i++) {
    if (uopdata->ops[i]==op) return i;
  }
  return -1;
}

/**
 * @param uop_idx
 * @return
 */
int isUserDefinedOp(int uop_idx) {
  return uop_idx!=-1;
}

/**
 */
void releaseUOpData();

int adDiffMPI_isActiveType(MPI_Datatype datatype) {
  return (datatype==MPI_FLOAT || datatype==MPI_REAL || datatype==MPI_REAL4
          || datatype==MPI_DOUBLE || datatype==MPI_DOUBLE_PRECISION || datatype==MPI_REAL8
          || datatype==MPI_REAL16
/* #ifdef AMPI_FORTRANCOMPATIBLE */
/*           || datatype==AMPI_ADOUBLE_PRECISION || datatype==AMPI_AREAL */
/* #endif */
          ) ;
}

/**
 * Register the info that the shadow communicator "dupComm"
 * has been created for the new communicator "comm"
 */
void adDiffMPI_addShadowComm(MPI_Comm comm, MPI_Comm dupComm) {
  struct AMPI_ShadowComm_list *newCell =
    (struct AMPI_ShadowComm_list *)malloc(sizeof(struct AMPI_ShadowComm_list)) ;
  newCell->next_p = adDiffMPI_shadowComm_list ;
  newCell->comm = comm;
  newCell->shadowComm = dupComm;
  adDiffMPI_shadowComm_list = newCell;
}

/**
 * Get the shadow communicator used to separate the communication graph of
 * (tangent-)diff variables from the communication graph of original variables
 */
MPI_Comm adDiffMPI_getShadowComm(MPI_Comm comm) {
  struct AMPI_ShadowComm_list * inShadowCommList = adDiffMPI_shadowComm_list ;
  while (inShadowCommList!=NULL && inShadowCommList->comm!=comm) {
    inShadowCommList = inShadowCommList->next_p ;
  }
  if (inShadowCommList) {
    return inShadowCommList->shadowComm ;
  } else {
    /* Nothing found about "comm": this is wrong!! fallback return comm */
    return comm ;
  }
}

/**
 * Removes the info about the shadow communicator associated to "comm".
 */
void adDiffMPI_delShadowComm(MPI_Comm comm) {
  struct AMPI_ShadowComm_list ** toinShadowCommList = &adDiffMPI_shadowComm_list ;
  while (*toinShadowCommList!=NULL && (*toinShadowCommList)->comm!=comm) {
    toinShadowCommList = &((*toinShadowCommList)->next_p) ;
  }
  if (*toinShadowCommList!=NULL) {
    struct AMPI_ShadowComm_list *cell = *toinShadowCommList ;
    toinShadowCommList = &(cell->next_p) ;
    free(cell);
  }
}

void adDiffMPI_getAdjointCount(int *count, MPI_Datatype datatype) {
  /* for now we keep the count as is but for example in vector mode one would have to multiply by vector length */
}

void* adDiffMPI_allocateTempBuf(int adjointCount, MPI_Datatype datatype, MPI_Comm comm) {
  size_t s=0;
  int dt_idx = derivedTypeIdx(datatype);
  if (datatype==MPI_FLOAT || datatype==MPI_REAL || datatype==MPI_REAL4)
    s=sizeof(float);
  else if (datatype==MPI_DOUBLE || datatype==MPI_DOUBLE_PRECISION || datatype==MPI_REAL8)
    s=sizeof(double);
  else if (datatype==MPI_REAL16)
    s=2*sizeof(double);
  else if (isDerivedType(dt_idx))
    s = getDTypeData()->p_extents[dt_idx];
  else
    MPI_Abort(comm, MPI_ERR_TYPE);
  return (void*)malloc(adjointCount*s);
}

/**
 * Increment the given buffer "target", which holds an adjoint variable,
 * with the given additional adjoint value found in "source".
 */
void adDiffMPI_incrementAdjoint(int adjointCount, MPI_Datatype datatype, MPI_Comm comm, void* target, void *source, void *idx) { 
  int dt_idx = derivedTypeIdx(datatype);
  if (isUserDefinedOp(dt_idx)) {
    derivedTypeData* dat = getDTypeData();
    MPI_Aint lb, extent ;
    MPI_Type_get_extent(datatype,&lb,&extent);
    MPI_Aint*   fieldOffsets = dat->arrays_of_displacements[dt_idx] ;
    int*   fieldBlocklengths = dat->arrays_of_blocklengths[dt_idx] ;
    MPI_Datatype* fieldTypes = dat->arrays_of_types[dt_idx] ;
    int nbfields = dat->counts[dt_idx] ;
    int i,j ;
    for (i=0 ; i<adjointCount ; ++i) {
      for (j=0 ; j<nbfields ; ++j) {
        adDiffMPI_incrementAdjoint(fieldBlocklengths[j], fieldTypes[j], comm,
                                   target+fieldOffsets[j], source+fieldOffsets[j], idx) ;
      }
      target += extent ;
      source += extent ;
    }
  } else if (datatype==MPI_FLOAT || datatype==MPI_REAL || datatype==MPI_REAL4) {
    float *vb = (float *)target ;
    float *nb = (float *)source ;
    int i ;
    for (i=0 ; i<adjointCount ; ++i) {
      *vb = *vb + *nb ;
      ++vb ;
      ++nb ;
    }
  } else if (datatype==MPI_DOUBLE || datatype==MPI_DOUBLE_PRECISION || datatype==MPI_REAL8) {
    double *vb = (double *)target ;
    double *nb = (double *)source ;
    int i ;
    for (i=0 ; i<adjointCount ; ++i) {
      *vb = *vb + *nb ;
      ++vb ;
      ++nb ;
    }
  } else {
    MPI_Abort(comm, MPI_ERR_TYPE);
  }
}

/**
 * Reset to zero the given buffer "target", which holds an adjoint variable.
 */
void adDiffMPI_nullifyAdjoint(int adjointCount, MPI_Datatype datatype, MPI_Comm comm,
                                void* target) {
  int dt_idx = derivedTypeIdx(datatype);
  if (isUserDefinedOp(dt_idx)) {
    derivedTypeData* dat = getDTypeData();
    MPI_Aint lb, extent ;
    MPI_Type_get_extent(datatype,&lb,&extent);
    MPI_Aint*   fieldOffsets = dat->arrays_of_displacements[dt_idx] ;
    int*   fieldBlocklengths = dat->arrays_of_blocklengths[dt_idx] ;
    MPI_Datatype* fieldTypes = dat->arrays_of_types[dt_idx] ;
    int nbfields = dat->counts[dt_idx] ;
    int i,j ;
    for (i=0 ; i<adjointCount ; ++i) {
      for (j=0 ; j<nbfields ; ++j) {
        adDiffMPI_nullifyAdjoint(fieldBlocklengths[j], fieldTypes[j], comm,
                                   target+fieldOffsets[j]) ;
      }
      target += extent ;
    }
  } else if (datatype==MPI_FLOAT || datatype==MPI_REAL || datatype==MPI_REAL4) {
    float *vb = (float *)target ;
    int i ;
    for (i=0 ; i<adjointCount ; ++i) {
      *vb = 0.0 ;
      ++vb ;
    }
  } else if (datatype==MPI_DOUBLE || datatype==MPI_DOUBLE_PRECISION || datatype==MPI_REAL8) {
    double *vb = (double *)target ;
    int i ;
    for (i=0 ; i<adjointCount ; ++i) {
      *vb = 0.0 ;
      ++vb ;
    }
  } else {
    MPI_Abort(comm, MPI_ERR_TYPE);
  }
}

// TODO: INITIALIZATION TO BE DONE SOMEWHERE !! :
/* init() { */
/*   MPI_Comm worldDup ; */
/*   int rc2 = MPI_Comm_dup(MPI_COMM_WORLD, &worldDup) ; */
/*   adDiffMPI_addShadowComm(MPI_COMM_WORLD, worldDup) ; */
/* } */

//############## TLS_, FW_, and BW_ DERIVATIVES OF MPI PRIMITIVES  ################
// (see Common/modified.c)

// GATHER:
int TLS_MPI_Gather(void *sendbuf, void *shadowsendbuf,
                   int sendcnt,
                   MPI_Datatype sendtype, MPI_Datatype shadowsendtype,
                   void *recvbuf, void *shadowrecvbuf,
                   int recvcnt,
                   MPI_Datatype recvtype, MPI_Datatype shadowrecvtype,
                   int root,
                   MPI_Comm comm) {
  int rc = MPI_Gather(sendbuf, sendcnt, sendtype, recvbuf, recvcnt, recvtype, root, comm) ;
  assert(rc==MPI_SUCCESS);
  rc = MPI_Gather(shadowsendbuf, sendcnt, shadowsendtype, shadowrecvbuf, recvcnt, shadowrecvtype, root, comm) ;
  return rc;
}

int FW_MPI_Gather(void *sendbuf, int sendcnt, MPI_Datatype sendtype,
                  void *recvbuf, int recvcnt, MPI_Datatype recvtype,
                  int root, MPI_Comm comm) {
  int rc=MPI_SUCCESS;
  int isInPlace=(sendbuf==MPI_IN_PLACE);
  int myRank, myCommSize;
  MPI_Comm_rank(comm, &myRank);
  MPI_Comm_size(comm, &myCommSize);
  if (!isInPlace && adDiffMPI_isActiveType(sendtype)!=adDiffMPI_isActiveType(recvtype)) {
    rc=MPI_Abort(comm, MPI_ERR_ARG);
  } else {
    rc=MPI_Gather(sendbuf,
		  sendcnt,
		  sendtype,
		  recvbuf,
		  recvcnt,
		  recvtype,
		  root,
		  comm);
    if (rc==MPI_SUCCESS && adDiffMPI_isActiveType(recvtype)) {
      pushInteger4(myRank==root?myCommSize:0) ;
    }
  }
  return rc;
}

int BW_MPI_Gather(void *sendbuf, int sendcnt, MPI_Datatype sendtype,
                  void *recvbuf, int recvcnt, MPI_Datatype recvtype,
                  int root, MPI_Comm comm) {
  void *idx=NULL;
  int rc=MPI_SUCCESS;
  int commSizeForRootOrNull, rTypeSize,i;
  popInteger4(&commSizeForRootOrNull);
  adDiffMPI_getAdjointCount(&sendcnt,sendtype);
  void *tempBuf = NULL ;
  if (sendcnt>0)
    tempBuf = adDiffMPI_allocateTempBuf(sendcnt,sendtype,comm) ;
  else if (commSizeForRootOrNull) 
    tempBuf = MPI_IN_PLACE;
  rc=MPI_Scatter(recvbuf,
		 recvcnt,
		 recvtype,
		 tempBuf,
		 sendcnt,
		 sendtype,
		 root,
		 comm);
  adDiffMPI_incrementAdjoint(sendcnt, sendtype, comm,
                             sendbuf, tempBuf, idx);
  if (commSizeForRootOrNull) {
    MPI_Type_size(recvtype,&rTypeSize);
    for (i=0;i<commSizeForRootOrNull;++i) { 
      if (! (i==root && sendcnt==0)) { /* don't nullify the segment if "in place" on root */
	void *recvbufSegment=(char*)recvbuf+(i*recvcnt*rTypeSize);
	adDiffMPI_nullifyAdjoint(recvcnt,recvtype,comm, recvbufSegment);
      }
    }
  }
  if (tempBuf!=MPI_IN_PLACE && tempBuf!=NULL) free(tempBuf);
  return rc;
}

// GATHERV:
//LLH-NOTE: argument shadowdispls is useless. Use displs instead.
int TLS_MPI_Gatherv(void *sendbuf, void *shadowsendbuf,
                    int sendcnt,
                    MPI_Datatype sendtype, MPI_Datatype shadowsendtype,
                    void *recvbuf, void *shadowrecvbuf,
                    int *recvcnts,
                    int *displs, int *shadowdispls,
                    MPI_Datatype recvtype, MPI_Datatype shadowrecvtype,
                    int root,
                    MPI_Comm comm) {
  int rc = MPI_Gatherv(sendbuf, sendcnt, sendtype, recvbuf, recvcnts, displs, recvtype, root, comm) ;
  assert(rc==MPI_SUCCESS);
  rc = MPI_Gatherv(shadowsendbuf, sendcnt, shadowsendtype, shadowrecvbuf, recvcnts, displs, shadowrecvtype, root, comm) ;
  return rc;
}

int FW_MPI_Gatherv(void *sendbuf, int sendcnt, MPI_Datatype sendtype,
                   void *recvbuf, int *recvcnts, int *displs, MPI_Datatype recvtype,
                  int root, MPI_Comm comm) {
  int rc=MPI_SUCCESS;
  int isInPlace=(sendbuf==MPI_IN_PLACE);
  int myRank, myCommSize;
  MPI_Comm_rank(comm, &myRank);
  MPI_Comm_size(comm, &myCommSize);
  if (!isInPlace && adDiffMPI_isActiveType(sendtype)!=adDiffMPI_isActiveType(recvtype)) {
    rc=MPI_Abort(comm, MPI_ERR_ARG);
  } else {
    rc=MPI_Gatherv(sendbuf,
                   sendcnt,
                   sendtype,
                   recvbuf,
                   recvcnts,
                   displs,
                   recvtype,
                   root,
                   comm);
    if (rc==MPI_SUCCESS && adDiffMPI_isActiveType(recvtype)) {
      pushInteger4(myRank==root?myCommSize:0) ;
    }
  }
  return rc;
}

//LLH-NOTE: Imperfect code: assumes non-NULL recvcnts and displs: refer to original AMPI.
int BW_MPI_Gatherv(void *sendbuf, int sendcnt, MPI_Datatype sendtype,
                   void *recvbuf, int *recvcnts, int *displs, MPI_Datatype recvtype,
                   int root, MPI_Comm comm) {
  void *idx=NULL;
  int rc=MPI_SUCCESS;
  int commSizeForRootOrNull, rTypeSize,i;
  popInteger4(&commSizeForRootOrNull);
  adDiffMPI_getAdjointCount(&sendcnt,sendtype);
  void *tempBuf = NULL ;
  if (sendcnt>0)
    tempBuf = adDiffMPI_allocateTempBuf(sendcnt,sendtype,comm) ;
  else if (commSizeForRootOrNull) 
    tempBuf = MPI_IN_PLACE;
  rc=MPI_Scatterv(recvbuf,
                  recvcnts,
                  displs,
                  recvtype,
                  tempBuf,
                  sendcnt,
                  sendtype,
                  root,
                  comm);
  adDiffMPI_incrementAdjoint(sendcnt, sendtype, comm,
                             sendbuf, tempBuf, idx);
  if (commSizeForRootOrNull) {
    MPI_Type_size(recvtype,&rTypeSize);
    for (i=0;i<commSizeForRootOrNull;++i) { 
      if (! (i==root && sendcnt==0)) { /* don't nullify the segment if "in place" on root */
	void *recvbufSegment=(char*)recvbuf+(displs[i]*rTypeSize);
	adDiffMPI_nullifyAdjoint(recvcnts[i],recvtype,comm, recvbufSegment);
      }
    }
  }
  if (tempBuf!=MPI_IN_PLACE && tempBuf!=NULL) free(tempBuf);
  return rc;
}

// SCATTERV:
//LLH-NOTE: argument shadowdispls is useless. Use displs instead.
int TLS_MPI_Scatterv(void *sendbuf, void *shadowsendbuf,
                     int *sendcnts,
                     int *displs, int *shadowdispls,
                     MPI_Datatype sendtype, MPI_Datatype shadowsendtype,
                     void *recvbuf, void *shadowrecvbuf,
                     int recvcnt,
                     MPI_Datatype recvtype, MPI_Datatype shadowrecvtype,
                     int root, MPI_Comm comm) {
  int rc = MPI_Scatterv(sendbuf, sendcnts, displs, sendtype, recvbuf, recvcnt, recvtype, root, comm) ;
  assert(rc==MPI_SUCCESS);
  rc = MPI_Scatterv(shadowsendbuf, sendcnts, displs, shadowsendtype, shadowrecvbuf, recvcnt, shadowrecvtype, root, comm) ;
  return rc;
}

int FW_MPI_Scatterv(void *sendbuf, int *sendcnts, int *displs, MPI_Datatype sendtype,
                     void *recvbuf, int recvcnt, MPI_Datatype recvtype,
                     int root, MPI_Comm comm) {
  int rc=MPI_SUCCESS;
  int myRank, myCommSize;
  int isInPlace=(recvbuf==MPI_IN_PLACE);
  MPI_Comm_rank(comm, &myRank);
  MPI_Comm_size(comm, &myCommSize);
  if (!isInPlace && adDiffMPI_isActiveType(sendtype)!=adDiffMPI_isActiveType(recvtype)) {
    rc=MPI_Abort(comm, MPI_ERR_ARG);
  }
  else {
    rc=MPI_Scatterv(sendbuf,
                    sendcnts,
                    displs,
                    sendtype,
                    recvbuf,
                    recvcnt,
                    recvtype,
                    root,
                    comm);
    if (rc==MPI_SUCCESS && adDiffMPI_isActiveType(sendtype)) {
      //? (*ourADTOOL_AMPI_FPCollection.push_CallCodeReserve_fp)(AMPI_SCATTERV,((myRank==root)?myCommSize:0)*2);
      pushInteger4(myRank==root?myCommSize:0) ;
    }
  }
  return rc;
}

int BW_MPI_Scatterv(void *sendbuf, int *sendcnts, int *displs, MPI_Datatype sendtype,
                    void *recvbuf, int recvcnt, MPI_Datatype recvtype,
                    int root, MPI_Comm comm) {
  int rc=MPI_SUCCESS;
  void *idx=NULL;
  int sendSize=0,i, typeSize;
  int myRank, commSizeForRootOrNull, *tempDispls;
  int *tSendCnts=sendcnts, *tDispls=displs;
  char tSendCntsFlag=0, tDisplsFlag=0;
  popInteger4(&commSizeForRootOrNull);
  if (tSendCnts==NULL && commSizeForRootOrNull>0) {
    tSendCnts=(int*)malloc(sizeof(int)*commSizeForRootOrNull);
    tSendCntsFlag=1;
  }
  if (tDispls==NULL && commSizeForRootOrNull>0) {
    tDispls=(int*)malloc(sizeof(int)*commSizeForRootOrNull);
    tDisplsFlag=1;
  }
  MPI_Comm_rank(comm, &myRank);
  tempDispls=(int*)malloc(sizeof(int)*commSizeForRootOrNull);
  for (i=0;i<commSizeForRootOrNull;++i) {
    tempDispls[i]=sendSize;
    sendSize+=tSendCnts[i];
  }
  void *tempBuf = NULL;
  if (commSizeForRootOrNull>0) tempBuf=adDiffMPI_allocateTempBuf(sendSize,sendtype,comm);
  rc=MPI_Gatherv(recvbuf,
                 recvcnt,
                 recvtype,
                 tempBuf,
                 tSendCnts,
                 tempDispls,
                 sendtype,
                 root,
                 comm);
  adDiffMPI_nullifyAdjoint(recvcnt,recvtype,comm,recvbuf);
  if (commSizeForRootOrNull>0) {
    MPI_Type_size(sendtype,&typeSize);
    for (i=0;i<commSizeForRootOrNull;++i) {
      if (! (i==root && recvcnt==0)) { /* don't increment the segment if "in place" on root */
        void* buf=(char*)sendbuf+(typeSize*tDispls[i]); /* <----------  very iffy! */
        void* sourceBuf=(char*)tempBuf+(typeSize*tempDispls[i]);
        adDiffMPI_incrementAdjoint(tSendCnts[i],
                                   sendtype,
                                   comm,
                                   buf,
                                   sourceBuf, idx);
      }
    }
    if (tempBuf!=NULL) free(tempBuf);
  }
  if (tempDispls) free((void*)tempDispls);
  if (tSendCntsFlag) free((void*)(tSendCnts));
  if (tDisplsFlag) free((void*)(tDispls));
  return rc;
}

//############## INTERFACE PROCEDURES CALLED FROM FORTRAN ################
// (see Common/ampi_f2c.c)

/*================================================*/
/*               gather                           */
/*================================================*/
void tls_mpi_gather_(void *sendbuf, void *shadowsendbuf,
                     int *sendcnt,
                     MPI_Fint *stypeF, MPI_Fint *shadowstypeF,
                     void *recvbuf, void *shadowrecvbuf,
                     int *recvcnt,
                     MPI_Fint *rtypeF, MPI_Fint *shadowrtypeF,
                     int *root, int *commF, int *err_code) {
  MPI_Datatype stype = MPI_Type_f2c(*stypeF) ;
  MPI_Datatype shadowstype = MPI_Type_f2c(*shadowstypeF) ;
  MPI_Datatype rtype = MPI_Type_f2c(*rtypeF) ;
  MPI_Datatype shadowrtype = MPI_Type_f2c(*shadowrtypeF) ;
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = TLS_MPI_Gather(sendbuf, shadowsendbuf,
                         *sendcnt,
                         stype, shadowstype,
                         recvbuf, shadowrecvbuf,
                         *recvcnt,
                         rtype, shadowrtype,
                         *root, commC) ;
}

void fw_mpi_gather_(void* sbuf, int *scount, MPI_Fint *stypeF,
                    void* rbuf, int *rcount, MPI_Fint *rtypeF,
                    int *root, int *commF, int* err_code) {
  MPI_Datatype stype = MPI_Type_f2c(*stypeF) ;
  MPI_Datatype rtype = MPI_Type_f2c(*rtypeF) ;
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = FW_MPI_Gather(sbuf, *scount, stype, rbuf, *rcount, rtype, *root, commC) ;
}

void bw_mpi_gather_(void* sbuf, int *scount, MPI_Fint *stypeF,
                    void* rbuf, int *rcount, MPI_Fint *rtypeF,
                    int *root, int *commF, int* err_code) {
  MPI_Datatype stype = MPI_Type_f2c(*stypeF) ;
  MPI_Datatype rtype = MPI_Type_f2c(*rtypeF) ;
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = BW_MPI_Gather(sbuf, *scount, stype, rbuf, *rcount, rtype, *root, commC) ;
}

/*================================================*/
/*               gatherv                          */
/*================================================*/
void tls_mpi_gatherv_(void *sendbuf, void *shadowsendbuf,
                      int *sendcnt,
                      MPI_Fint *stypeF, MPI_Fint *shadowstypeF,
                      void *recvbuf, void *shadowrecvbuf,
                      int *recvcnts,
                      int *displs, int *shadowdispls,
                      MPI_Fint *rtypeF, MPI_Fint *shadowrtypeF,
                      int *root, int *commF, int *err_code) {
  MPI_Datatype stype = MPI_Type_f2c(*stypeF) ;
  MPI_Datatype shadowstype = MPI_Type_f2c(*shadowstypeF) ;
  MPI_Datatype rtype = MPI_Type_f2c(*rtypeF) ;
  MPI_Datatype shadowrtype = MPI_Type_f2c(*shadowrtypeF) ;
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = TLS_MPI_Gatherv(sendbuf, shadowsendbuf,
                              *sendcnt,
                              stype, shadowstype,
                              recvbuf, shadowrecvbuf,
                              recvcnts,
                              displs, shadowdispls,
                              rtype, shadowrtype,
                              *root, commC) ;
}

void fw_mpi_gatherv_(void* sbuf, int *scount, MPI_Fint *stypeF,
                     void* rbuf, int *rcount, int *displs, MPI_Fint *rtypeF,
                     int *root, int *commF, int* err_code) {
  MPI_Datatype stype = MPI_Type_f2c(*stypeF) ;
  MPI_Datatype rtype = MPI_Type_f2c(*rtypeF) ;
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = FW_MPI_Gatherv(sbuf, *scount, stype, rbuf, rcount, displs, rtype, *root, commC) ;
}

void bw_mpi_gatherv_(void* sbuf, int *scount, MPI_Fint *stypeF,
                     void* rbuf, int *rcount, int *displs, MPI_Fint *rtypeF,
                     int *root, int *commF, int* err_code) {
  MPI_Datatype stype = MPI_Type_f2c(*stypeF) ;
  MPI_Datatype rtype = MPI_Type_f2c(*rtypeF) ;
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = BW_MPI_Gatherv(sbuf, *scount, stype, rbuf, rcount, displs, rtype, *root, commC) ;
}

/*================================================*/
/*               scatterv                         */
/*================================================*/
void tls_mpi_scatterv_(void *sendbuf, void *shadowsendbuf,
                       int *sendcnts,
                       int *displs, int *shadowdispls,
                       MPI_Fint *stypeF, MPI_Fint *shadowstypeF,
                       void *recvbuf, void *shadowrecvbuf,
                       int *recvcnt,
                       MPI_Fint *rtypeF, MPI_Fint *shadowrtypeF,
                       int *root, int *commF, int *err_code) {
  MPI_Datatype stype = MPI_Type_f2c(*stypeF) ;
  MPI_Datatype shadowstype = MPI_Type_f2c(*shadowstypeF) ;
  MPI_Datatype rtype = MPI_Type_f2c(*rtypeF) ;
  MPI_Datatype shadowrtype = MPI_Type_f2c(*shadowrtypeF) ;
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = TLS_MPI_Scatterv(sendbuf, shadowsendbuf,
                           sendcnts,
                           displs, shadowdispls,
                           stype, shadowstype,
                           recvbuf, shadowrecvbuf,
                           *recvcnt,
                           rtype, shadowrtype,
                           *root, commC) ;
}

void fw_mpi_scatterv_(void* sbuf, int *scounts, int *displs, MPI_Fint *stypeF,
                      void* rbuf, int *rcount, MPI_Fint *rtypeF,
                      int *root, int *commF, int* err_code) {
  MPI_Datatype stype = MPI_Type_f2c(*stypeF) ;
  MPI_Datatype rtype = MPI_Type_f2c(*rtypeF) ;
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = FW_MPI_Scatterv(sbuf, scounts, displs, stype, rbuf, *rcount, rtype, *root, commC) ;
}

void bw_mpi_scatterv_(void* sbuf, int *scounts, int *displs, MPI_Fint *stypeF,
                      void* rbuf, int *rcount, MPI_Fint *rtypeF,
                      int *root, int *commF, int* err_code) {
  MPI_Datatype stype = MPI_Type_f2c(*stypeF) ;
  MPI_Datatype rtype = MPI_Type_f2c(*rtypeF) ;
  MPI_Comm commC = MPI_Comm_f2c( *commF ) ;
  *err_code = BW_MPI_Scatterv(sbuf, scounts, displs, stype, rbuf, *rcount, rtype, *root, commC) ;
}

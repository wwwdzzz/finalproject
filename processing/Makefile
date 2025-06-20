SOURCE=mainpro.c
TARGET=mainpro

PETSC_DIR := /home/wdz/petsc

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

mainpro: mainpro.o
	${CLINKER} -o $@ $< ${PETSC_KSP_LIB}


SOURCE=mainve.c
TARGET=mainve

PETSC_DIR := /home/wdz/petsc

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

mainve: mainve.o
	${CLINKER} -o $@ $< ${PETSC_KSP_LIB}


SOURCE=main.c
TARGET=main

PETSC_DIR := /home/wdz/petsc

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

main: main.o
	${CLINKER} -o $@ $< ${PETSC_KSP_LIB}


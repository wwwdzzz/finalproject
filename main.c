#include <petscksp.h>
#include <petscmat.h>
#include <petscvec.h>
#include <petscsys.h>
#include <math.h>
#include <time.h>

int main(int argc, char **argv)
{
   double timestart=clock();
    PetscInitialize(&argc, &argv, NULL, NULL);
    
    PetscMPIInt size, rank;
    MPI_Comm_size(PETSC_COMM_WORLD, &size);
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    
    PetscInt N = 10; 
    PetscInt max_iter = 100; 
    PetscReal tol = 1.0e-6; 
    PetscBool monitor = PETSC_FALSE; 

    PetscOptionsGetInt(NULL, NULL, "-N", &N, NULL);
    PetscOptionsGetInt(NULL, NULL, "-max_iter", &max_iter, NULL);
    PetscOptionsGetReal(NULL, NULL, "-tol", &tol, NULL);
    PetscOptionsGetBool(NULL, NULL, "-monitor", &monitor, NULL);
    
    if (rank == 0) {
        printf("Running inverse power iteration for %d x %d matrix\n", N, N);
        printf("Max iterations: %d, Tolerance: %g\n", max_iter, tol);
    }
    
    Mat A;
    MatCreate(PETSC_COMM_WORLD, &A);
    MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, N, N);
    MatSetFromOptions(A);
    MatSetUp(A);
    
    PetscInt i, Istart, Iend;
    MatGetOwnershipRange(A, &Istart, &Iend);
    
    for (i = Istart; i < Iend; i++) {
        PetscScalar v = 2.0;
        MatSetValues(A, 1, &i, 1, &i, &v, INSERT_VALUES);
        
        if (i > 0) {
            PetscInt j = i - 1;
            PetscScalar v = -1.0;
            MatSetValues(A, 1, &i, 1, &j, &v, INSERT_VALUES);
        }
        
        if (i < N - 1) {
            PetscInt j = i + 1;
            PetscScalar v = -1.0;
            MatSetValues(A, 1, &i, 1, &j, &v, INSERT_VALUES);
        }
    }
    
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
    
    KSP ksp;
    KSPCreate(PETSC_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, A, A);
 
    KSPSetFromOptions(ksp);
    
    if (monitor) {
        KSPMonitorSet(ksp, KSPMonitorDefault, NULL, NULL);
    }
    
    Vec z_prev, z_curr, y;
    VecCreate(PETSC_COMM_WORLD, &z_prev);
    VecCreate(PETSC_COMM_WORLD, &z_curr);
    VecCreate(PETSC_COMM_WORLD, &y);
    VecSetSizes(z_prev, PETSC_DECIDE, N);
    VecSetSizes(z_curr, PETSC_DECIDE, N);
    VecSetSizes(y, PETSC_DECIDE, N);
    VecSetFromOptions(z_prev);
    VecSetFromOptions(z_curr);
    VecSetFromOptions(y);
    
    PetscScalar one = 1.0;
    PetscScalar zero = 0.0;
    if (Istart == 0) {
        VecSetValue(z_prev, 0, one, INSERT_VALUES);
    }
    for (i = 1; i < N; i++) {
        VecSetValue(z_prev, i, zero, INSERT_VALUES);
    }
    VecAssemblyBegin(z_prev);
    VecAssemblyEnd(z_prev);

    PetscReal norm_prev = 0.0, norm_curr = 0.0;
    PetscReal lambda = 0.0;
    PetscInt k;
    PetscBool converged = PETSC_FALSE;
    
    for (k = 0; k < max_iter; k++) {
        KSPSolve(ksp, z_prev, y);

        VecNorm(y, NORM_2, &norm_curr);

        VecCopy(y, z_curr);
        VecScale(z_curr, 1.0/norm_curr);

        if (k > 0 && PetscAbsReal(norm_curr - norm_prev) < tol) {
            converged = PETSC_TRUE;
            break;
        }

        VecCopy(z_curr, z_prev);
        norm_prev = norm_curr;
    }

    if (converged) {
        Vec Ay;
        VecDuplicate(y, &Ay);
        MatMult(A, y, Ay);
        VecDot(y, Ay, &lambda);
        lambda = 1.0 / lambda;
        
        if (rank == 0) {
            printf("Converged after %d iterations\n", k);
            printf("Smallest eigenvalue estimate: %g\n", lambda);
        }
    } else {
        if (rank == 0) {
            printf("Did not converge after %d iterations\n", max_iter);
        }
    }

    VecDestroy(&z_prev);
    VecDestroy(&z_curr);
    VecDestroy(&y);
    MatDestroy(&A);
    KSPDestroy(&ksp);
    
    PetscFinalize();
    double timeend=clock();
    
    double cputimeused=((double)(timeend-timestart))/CLOCKS_PER_SEC;
    printf("cpu timeused %f s\n",cputimeused);
    return 0;
}

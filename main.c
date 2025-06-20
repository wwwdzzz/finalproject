#include <petscdm.h>
#include <petscdmda.h>
#include <petscmat.h>
#include <petscvec.h>
#include <petscksp.h>
#include <math.h>
#include <time.h>

typedef struct {
    PetscReal kappa;
    PetscReal rho;
    PetscReal c;
    PetscReal dx;
    PetscReal dt;
    PetscReal T_final;
    PetscInt  method;
    PetscInt  nx;
    DM        da;
    Mat       A;
    KSP       ksp;
} AppCtx;

PetscErrorCode Initialize(AppCtx* user) {
    PetscFunctionBeginUser;
    
    if (user->method == 1) {
        DMCreateMatrix(user->da, &user->A);
        
        PetscInt i, xs, xn;
        DMDAGetCorners(user->da, &xs, NULL, NULL, &xn, NULL, NULL);
        PetscReal coeff = user->kappa / (user->rho * user->c * user->dx * user->dx);
        
        for (i = xs; i < xs+xn; i++) {
            PetscInt row = i;
            PetscScalar value;
            
            if (i == 0 || i == user->nx-1) {
                MatSetValue(user->A, row, row, 1.0, INSERT_VALUES);
                continue;
            }
            
            value = 1.0 + 2.0 * user->dt * coeff;
            MatSetValue(user->A, row, row, value, INSERT_VALUES);
            
            value = -user->dt * coeff;
            MatSetValue(user->A, row, row-1, value, INSERT_VALUES);
            MatSetValue(user->A, row, row+1, value, INSERT_VALUES);
        }
        
        MatAssemblyBegin(user->A, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(user->A, MAT_FINAL_ASSEMBLY);
        
        KSPCreate(PETSC_COMM_WORLD, &user->ksp);
        KSPSetOperators(user->ksp, user->A, user->A);
        KSPSetFromOptions(user->ksp);
    }
    
    PetscFunctionReturn(0);
}

PetscErrorCode SetInitialConditions(Vec U, AppCtx* user) {
    PetscFunctionBeginUser;
    PetscScalar *u;
    VecGetArray(U, &u);
    
    PetscInt i, xs, xn;
    DMDAGetCorners(user->da, &xs, NULL, NULL, &xn, NULL, NULL);
    
    for (i = xs; i < xs+xn; i++) {
        PetscReal x = i * user->dx;
        u[i] = sin(M_PI * x);
    }
    
    VecRestoreArray(U, &u);
    PetscFunctionReturn(0);
}

PetscErrorCode ExplicitEulerStep(Vec U, Vec Unew, AppCtx* user) {
    PetscFunctionBeginUser;
    PetscScalar *u, *unew;
    VecGetArray(U, &u);
    VecGetArray(Unew, &unew);
    
    PetscInt i, xs, xn;
    DMDAGetCorners(user->da, &xs, NULL, NULL, &xn, NULL, NULL);
    PetscReal coeff = user->kappa / (user->rho * user->c * user->dx * user->dx);
    
    if (xs == 0) unew[0] = 0.0;
    if (xs + xn == user->nx) unew[user->nx-1] = 0.0;
    
    for (i = xs; i < xs+xn; i++) {
        if (i > 0 && i < user->nx-1) {
            unew[i] = u[i] + user->dt * coeff * (u[i-1] - 2.0*u[i] + u[i+1]);
        }
    }
    
    VecRestoreArray(U, &u);
    VecRestoreArray(Unew, &unew);
    
    DMLocalToGlobalBegin(user->da, Unew, INSERT_VALUES, Unew);
    DMLocalToGlobalEnd(user->da, Unew, INSERT_VALUES, Unew);
    
    PetscFunctionReturn(0);
}

PetscErrorCode ImplicitEulerStep(Vec U, Vec Unew, AppCtx* user) {
    PetscFunctionBeginUser;
    VecCopy(U, Unew);
    KSPSolve(user->ksp, U, Unew);
    PetscFunctionReturn(0);
}

int main(int argc, char **argv) {
    AppCtx         user;
    Vec            U, Unew;
    PetscInt       nsteps;
    PetscMPIInt    rank;
    PetscLogDouble t1, t2;
    
    PetscInitialize(&argc, &argv, NULL, NULL);
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    
    user.kappa = 1.0;
    user.rho = 1.0;
    user.c = 1.0;
    user.method = 1;
    user.nx = 100;
    user.dt = 0.001;
    user.T_final = 0.1;
    
    PetscOptionsGetInt(NULL, NULL, "-nx", &user.nx, NULL);
    PetscOptionsGetReal(NULL, NULL, "-dt", &user.dt, NULL);
    PetscOptionsGetReal(NULL, NULL, "-T", &user.T_final, NULL);
    PetscOptionsGetInt(NULL, NULL, "-method", &user.method, NULL);
    
    user.dx = 1.0 / (user.nx - 1);
    nsteps = (PetscInt)(user.T_final / user.dt) + 1;
    
    DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_GHOSTED, user.nx, 1, 1, NULL, &user.da);
    DMSetFromOptions(user.da);
    DMSetUp(user.da);
    
    Initialize(&user);
    
    // 修正点：使用.操作符而不是->，因为user是局部变量
    DMCreateGlobalVector(user.da, &U);
    DMCreateGlobalVector(user.da, &Unew);
    SetInitialConditions(U, &user);
    
    PetscTime(&t1);
    for (PetscInt step = 0; step < nsteps; step++) {
        if (user.method == 0) {
            ExplicitEulerStep(U, Unew, &user);
        } else {
            ImplicitEulerStep(U, Unew, &user);
        }
        VecSwap(U, Unew);
        
        if (step % 100 == 0 && rank == 0) {
            PetscPrintf(PETSC_COMM_SELF, "Step %d/%d, Time %.4f/%.4f\n", 
                       step, nsteps, step*user.dt, user.T_final);
        }
    }
    PetscTime(&t2);
    
    if (rank == 0) {
        PetscPrintf(PETSC_COMM_SELF, "Completed %d steps in %.4f seconds\n", 
                   nsteps, t2-t1);
    }
    
    PetscViewer viewer;
    PetscViewerASCIIOpen(PETSC_COMM_WORLD, "solution.txt", &viewer);
    VecView(U, viewer);
    PetscViewerDestroy(&viewer);
    
    VecDestroy(&U);
    VecDestroy(&Unew);
    if (user.method == 1) {
        MatDestroy(&user.A);
        KSPDestroy(&user.ksp);
    }
    DMDestroy(&user.da);
    PetscFinalize();
    return 0;
}

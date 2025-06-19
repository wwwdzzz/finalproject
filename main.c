#include <petscdm.h>
#include <petscdmda.h>
#include <petscts.h>
#include <petscksp.h>
#include <math.h>

typedef struct {
    PetscReal kappa;     // Thermal conductivity
    PetscReal rho;       // Density
    PetscReal c;         // Heat capacity
    PetscReal dx;        // Spatial step size
    PetscReal dt;        // Time step size
    PetscReal T_final;   // Final time
    PetscInt  method;    // 0=explicit, 1=implicit
    PetscInt  nx;        // Number of grid points (新增)
} AppCtx;

PetscErrorCode InitialConditions(DM da, Vec U, AppCtx *user)
{
    PetscFunctionBeginUser;
    PetscReal **u;
    PetscInt i, xs, xn;
    
    DMDAVecGetArray(da, U, &u);
    DMDAGetCorners(da, &xs, NULL, NULL, &xn, NULL, NULL);
    
    for (i = xs; i < xs+xn; i++) {
        PetscReal x = i * user->dx;
        u[i][0] = sin(M_PI * x);  // Initial condition: u(x,0) = sin(πx)
    }
    
    DMDAVecRestoreArray(da, U, &u);
    PetscFunctionReturn(0);
}

PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec U, Vec F, void *ctx)
{
    AppCtx         *user = (AppCtx*)ctx;
    DM             da;
    PetscReal      **u, **f;
    PetscInt       i, xs, xn;
    PetscReal      coeff = user->kappa / (user->rho * user->c * user->dx * user->dx);
    
    TSGetDM(ts, &da);
    DMDAVecGetArray(da, U, &u);
    DMDAVecGetArray(da, F, &f);
    DMDAGetCorners(da, &xs, NULL, NULL, &xn, NULL, NULL);
    
    // Left boundary (Dirichlet)
    if (xs == 0) {
        f[0][0] = 0.0;  // u(0,t) = 0
        xs++;
        xn--;
    }
    
    // Right boundary (Dirichlet)
    if (xs + xn == user->nx) {
        f[user->nx-1][0] = 0.0;  // u(1,t) = 0
        xn--;
    }
    
    // Interior points
    for (i = xs; i < xs+xn; i++) {
        f[i][0] = coeff * (u[i-1][0] - 2.0*u[i][0] + u[i+1][0]);
    }
    
    DMDAVecRestoreArray(da, U, &u);
    DMDAVecRestoreArray(da, F, &f);
    PetscFunctionReturn(0);
}

PetscErrorCode IFunction(TS ts, PetscReal t, Vec U, Vec Udot, Vec F, void *ctx)
{
    AppCtx         *user = (AppCtx*)ctx;
    DM             da;
    PetscReal      **u, **udot, **f;
    PetscInt       i, xs, xn;
    PetscReal      coeff = user->kappa / (user->rho * user->c * user->dx * user->dx);
    
    TSGetDM(ts, &da);
    DMDAVecGetArray(da, U, &u);
    DMDAVecGetArray(da, Udot, &udot);
    DMDAVecGetArray(da, F, &f);
    DMDAGetCorners(da, &xs, NULL, NULL, &xn, NULL, NULL);
    
    // Left boundary (Dirichlet)
    if (xs == 0) {
        f[0][0] = u[0][0];  // u(0,t) = 0
        xs++;
        xn--;
    }
    
    // Right boundary (Dirichlet)
    if (xs + xn == user->nx) {
        f[user->nx-1][0] = u[user->nx-1][0];  // u(1,t) = 0
        xn--;
    }
    
    // Interior points
    for (i = xs; i < xs+xn; i++) {
        f[i][0] = udot[i][0] - coeff * (u[i-1][0] - 2.0*u[i][0] + u[i+1][0]);
    }
    
    DMDAVecRestoreArray(da, U, &u);
    DMDAVecRestoreArray(da, Udot, &udot);
    DMDAVecRestoreArray(da, F, &f);
    PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
    DM             da;
    TS             ts;
    Vec            U;
    AppCtx         user;
    PetscInt       nx = 100;  // Default number of grid points
    PetscReal      dt = 0.001;
    PetscReal      T_final = 0.1;
    PetscMPIInt    size;
    
    PetscInitialize(&argc, &argv, NULL, NULL);
    MPI_Comm_size(PETSC_COMM_WORLD, &size);
    
    // Set default parameters
    user.kappa = 1.0;
    user.rho = 1.0;
    user.c = 1.0;
    user.method = 1;  // Default to implicit
    
    // Parse command line options
    PetscOptionsGetInt(NULL, NULL, "-nx", &nx, NULL);
    PetscOptionsGetReal(NULL, NULL, "-dt", &dt, NULL);
    PetscOptionsGetReal(NULL, NULL, "-T", &T_final, NULL);
    PetscOptionsGetInt(NULL, NULL, "-method", &user.method, NULL);
    PetscOptionsGetReal(NULL, NULL, "-kappa", &user.kappa, NULL);
    PetscOptionsGetReal(NULL, NULL, "-rho", &user.rho, NULL);
    PetscOptionsGetReal(NULL, NULL, "-c", &user.c, NULL);
    
    user.dx = 1.0 / (nx - 1);
    user.dt = dt;
    user.T_final = T_final;
    user.nx = nx;  // 初始化nx
    
    // Create distributed array
    DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, nx, 1, 1, NULL, &da);
    DMSetFromOptions(da);
    DMSetUp(da);
    
    // Create solution vector
    DMCreateGlobalVector(da, &U);
    InitialConditions(da, U, &user);
    
    // Create timestepper
    TSCreate(PETSC_COMM_WORLD, &ts);
    TSSetDM(ts, da);
    TSSetProblemType(ts, TS_LINEAR);
    TSSetType(ts, TSEULER);
    TSSetMaxSteps(ts, (PetscInt)(T_final / dt) + 1);
    TSSetTimeStep(ts, dt);
    TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP);
    
    if (user.method == 0) {  // Explicit
        TSSetRHSFunction(ts, NULL, RHSFunction, &user);
    } else {  // Implicit
        TSSetIFunction(ts, NULL, IFunction, &user);
        Mat A;
        MatCreate(da, &A);
        MatSetFromOptions(A);
        TSSetIJacobian(ts, A, A, NULL, NULL);
        MatDestroy(&A);
    }
    
    TSSetFromOptions(ts);
    TSSolve(ts, U);
    
    // Output final solution
    PetscViewer viewer;
    PetscViewerASCIIOpen(PETSC_COMM_WORLD, "solution.txt", &viewer);
    VecView(U, viewer);
    PetscViewerDestroy(&viewer);
    
    // Cleanup
    VecDestroy(&U);
    TSDestroy(&ts);
    DMDestroy(&da);
    PetscFinalize();
    return 0;
    }

#include <petscdm.h>
#include <petscdmda.h>
#include <petscmat.h>
#include <petscvec.h>
#include <petscksp.h>
#include <math.h>
#include <time.h>

typedef struct {
    PetscReal kappa;     // Thermal conductivity
    PetscReal rho;       // Density
    PetscReal c;         // Heat capacity
    PetscReal dx, dy;    // Spatial step sizes
    PetscReal dt;        // Time step size
    PetscReal T_final;   // Final time
    PetscInt  method;    // 0=explicit, 1=implicit
    PetscInt  nx, ny;    // Grid dimensions
    DM        da;        // Distributed array context
    Mat       A;         // Matrix for implicit method
    KSP       ksp;       // Linear solver context
    PetscReal bc_values[4]; // Boundary condition values
    PetscInt  bc_types[4];  // Boundary condition types
} AppCtx;

PetscErrorCode Initialize(AppCtx* user) {
    PetscFunctionBeginUser;
    
    if (user->method == 1) {
        DMCreateMatrix(user->da, &user->A);
        
        PetscInt i, j, xs, ys, xn, yn;
        DMDAGetCorners(user->da, &xs, &ys, NULL, &xn, &yn, NULL);
        
        PetscReal coeff_x = user->kappa * user->dt / (user->rho * user->c * user->dx * user->dx);
        PetscReal coeff_y = user->kappa * user->dt / (user->rho * user->c * user->dy * user->dy);
        
        for (j = ys; j < ys+yn; j++) {
            for (i = xs; i < xs+xn; i++) {
                PetscInt row = i + j * user->nx;
                PetscInt col[5];
                PetscScalar values[5];
                PetscInt ncols = 0;
                
                // Left boundary (Dirichlet)
                if (i == 0) {
                    // Set diagonal to 1 and right-hand side will set the value
                    MatSetValue(user->A, row, row, 1.0, INSERT_VALUES);
                    continue;
                }
                
                // Right boundary (Dirichlet)
                if (i == user->nx-1) {
                    // Set diagonal to 1 and right-hand side will set the value
                    MatSetValue(user->A, row, row, 1.0, INSERT_VALUES);
                    continue;
                }
                
                // Bottom boundary (Neumann: ∂u/∂y = 0)
                if (j == 0) {
                    // Use ghost point: u_{i,-1} = u_{i,1}
                    // So the equation becomes (u_{i,1} - u_{i,-1})/(2dy) = 0
                    // Which simplifies to the standard central difference
                    col[ncols] = row;           // u_{i,0}
                    values[ncols] = 1.0 + 2.0*coeff_x + 2.0*coeff_y;
                    ncols++;
                    col[ncols] = row - 1;       // u_{i-1,0}
                    values[ncols] = -coeff_x;
                    ncols++;
                    col[ncols] = row + 1;       // u_{i+1,0}
                    values[ncols] = -coeff_x;
                    ncols++;
                    col[ncols] = row + user->nx; // u_{i,1} (top neighbor)
                    values[ncols] = -2.0*coeff_y; // Because u_{i,-1} = u_{i,1}
                    ncols++;
                    MatSetValues(user->A, 1, &row, ncols, col, values, INSERT_VALUES);
                    continue;
                }
                
                // Top boundary (Neumann: ∂u/∂y = 0)
                if (j == user->ny-1) {
                    // Similar to bottom boundary
                    col[ncols] = row;           // u_{i,ny-1}
                    values[ncols] = 1.0 + 2.0*coeff_x + 2.0*coeff_y;
                    ncols++;
                    col[ncols] = row - 1;       // u_{i-1,ny-1}
                    values[ncols] = -coeff_x;
                    ncols++;
                    col[ncols] = row + 1;       // u_{i+1,ny-1}
                    values[ncols] = -coeff_x;
                    ncols++;
                    col[ncols] = row - user->nx; // u_{i,ny-2} (bottom neighbor)
                    values[ncols] = -2.0*coeff_y; // Because u_{i,ny} = u_{i,ny-2}
                    ncols++;
                    MatSetValues(user->A, 1, &row, ncols, col, values, INSERT_VALUES);
                    continue;
                }
                
                // Interior points
                col[ncols] = row;
                values[ncols] = 1.0 + 2.0*coeff_x + 2.0*coeff_y;
                ncols++;
                col[ncols] = row - 1;
                values[ncols] = -coeff_x;
                ncols++;
                col[ncols] = row + 1;
                values[ncols] = -coeff_x;
                ncols++;
                col[ncols] = row - user->nx;
                values[ncols] = -coeff_y;
                ncols++;
                col[ncols] = row + user->nx;
                values[ncols] = -coeff_y;
                ncols++;
                
                MatSetValues(user->A, 1, &row, ncols, col, values, INSERT_VALUES);
            }
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
    PetscScalar **u;
    DMDAVecGetArray(user->da, U, &u);
    
    PetscInt i, j, xs, ys, xn, yn;
    DMDAGetCorners(user->da, &xs, &ys, NULL, &xn, &yn, NULL);
    
    for (j = ys; j < ys+yn; j++) {
        PetscReal y = j * user->dy;
        for (i = xs; i < xs+xn; i++) {
            PetscReal x = i * user->dx;
            u[j][i] = (1.0 - x) + 0.1 * sin(2*M_PI*x) * sin(2*M_PI*y);
        }
    }
    
    DMDAVecRestoreArray(user->da, U, &u);
    PetscFunctionReturn(0);
}

PetscErrorCode ApplyBoundaryConditions(Vec U, AppCtx* user) {
    PetscFunctionBeginUser;
    PetscScalar **u;
    DMDAVecGetArray(user->da, U, &u);
    
    PetscInt i, j, xs, ys, xn, yn;
    DMDAGetCorners(user->da, &xs, &ys, NULL, &xn, &yn, NULL);
    
    // Left boundary (Dirichlet)
    if (xs == 0) {
        for (j = ys; j < ys+yn; j++) {
            u[j][0] = user->bc_values[0];
        }
    }
    
    // Right boundary (Dirichlet)
    if (xs + xn == user->nx) {
        for (j = ys; j < ys+yn; j++) {
            u[j][user->nx-1] = user->bc_values[1];
        }
    }
    
    // Bottom boundary (Neumann: ∂u/∂y = 0)
    if (ys == 0) {
        for (i = xs; i < xs+xn; i++) {
            if (i > 0 && i < user->nx-1) {
                u[0][i] = u[1][i];
            }
        }
    }
    
    // Top boundary (Neumann: ∂u/∂y = 0)
    if (ys + yn == user->ny) {
        for (i = xs; i < xs+xn; i++) {
            if (i > 0 && i < user->nx-1) {
                u[user->ny-1][i] = u[user->ny-2][i];
            }
        }
    }
    
    DMDAVecRestoreArray(user->da, U, &u);
    PetscFunctionReturn(0);
}

PetscErrorCode ExplicitEulerStep(Vec U, Vec Unew, AppCtx* user) {
    PetscFunctionBeginUser;
    PetscScalar **u, **unew;
    DMDAVecGetArray(user->da, U, &u);
    DMDAVecGetArray(user->da, Unew, &unew);
    
    PetscInt i, j, xs, ys, xn, yn;
    DMDAGetCorners(user->da, &xs, &ys, NULL, &xn, &yn, NULL);
    
    PetscReal coeff_x = user->kappa * user->dt / (user->rho * user->c * user->dx * user->dx);
    PetscReal coeff_y = user->kappa * user->dt / (user->rho * user->c * user->dy * user->dy);
    
    VecCopy(U, Unew);
    
    for (j = PetscMax(ys,1); j < PetscMin(ys+yn,user->ny-1); j++) {
        for (i = PetscMax(xs,1); i < PetscMin(xs+xn,user->nx-1); i++) {
            unew[j][i] = u[j][i] + 
                coeff_x * (u[j][i-1] - 2.0*u[j][i] + u[j][i+1]) +
                coeff_y * (u[j-1][i] - 2.0*u[j][i] + u[j+1][i]);
        }
    }
    
    DMDAVecRestoreArray(user->da, U, &u);
    DMDAVecRestoreArray(user->da, Unew, &unew);
    
    DMLocalToLocalBegin(user->da, Unew, INSERT_VALUES, Unew);
    DMLocalToLocalEnd(user->da, Unew, INSERT_VALUES, Unew);
    
    ApplyBoundaryConditions(Unew, user);
    
    PetscFunctionReturn(0);
}

PetscErrorCode ImplicitEulerStep(Vec U, Vec Unew, AppCtx* user) {
    PetscFunctionBeginUser;
    
    Vec b;
    VecDuplicate(U, &b);
    VecCopy(U, b);
    
    // Apply boundary conditions to right-hand side
    PetscScalar **b_array;
    DMDAVecGetArray(user->da, b, &b_array);
    
    PetscInt xs, ys, xn, yn;
    DMDAGetCorners(user->da, &xs, &ys, NULL, &xn, &yn, NULL);
    
    // Left boundary (Dirichlet)
    if (xs == 0) {
        for (PetscInt j = ys; j < ys+yn; j++) {
            b_array[j][0] = user->bc_values[0]; // Set to prescribed value
        }
    }
    
    // Right boundary (Dirichlet)
    if (xs + xn == user->nx) {
        for (PetscInt j = ys; j < ys+yn; j++) {
            b_array[j][user->nx-1] = user->bc_values[1]; // Set to prescribed value
        }
    }
    
    // For Neumann boundaries, no need to modify b as it's handled in the matrix
    
    DMDAVecRestoreArray(user->da, b, &b_array);
    
    // Solve the linear system: A*Unew = b
    KSPSolve(user->ksp, b, Unew);
    
    // Ensure boundary conditions are satisfied (important for parallel runs)
    ApplyBoundaryConditions(Unew, user);
    
    VecDestroy(&b);
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
    
    // Default parameters
    user.kappa = 1.0;
    user.rho = 1.0;
    user.c = 1.0;
    user.method = 1;
    user.nx = 20;  // Smaller grid for testing
    user.ny = 20;
    user.dt = 0.0001;
    user.T_final = 0.01; // Shorter time for testing
    
    // Boundary conditions
    user.bc_values[0] = 1.0;  // Left
    user.bc_values[1] = 0.0;  // Right
    user.bc_values[2] = 0.0;  // Bottom (unused)
    user.bc_values[3] = 0.0;  // Top (unused)
    
    user.bc_types[0] = 0;     // Left: Dirichlet
    user.bc_types[1] = 0;     // Right: Dirichlet
    user.bc_types[2] = 1;     // Bottom: Neumann
    user.bc_types[3] = 1;     // Top: Neumann
    
    PetscOptionsGetInt(NULL, NULL, "-nx", &user.nx, NULL);
    PetscOptionsGetInt(NULL, NULL, "-ny", &user.ny, NULL);
    PetscOptionsGetReal(NULL, NULL, "-dt", &user.dt, NULL);
    PetscOptionsGetReal(NULL, NULL, "-T", &user.T_final, NULL);
    PetscOptionsGetInt(NULL, NULL, "-method", &user.method, NULL);
    
    user.dx = 1.0 / (user.nx - 1);
    user.dy = 1.0 / (user.ny - 1);
    nsteps = (PetscInt)(user.T_final / user.dt) + 1;
    
    DMDACreate2d(PETSC_COMM_WORLD, 
                DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED,
                DMDA_STENCIL_STAR,
                user.nx, user.ny,
                PETSC_DECIDE, PETSC_DECIDE,
                1, 1,
                NULL, NULL,
                &user.da);
    DMSetFromOptions(user.da);
    DMSetUp(user.da);
    
    Initialize(&user);
    DMCreateGlobalVector(user.da, &U);
    DMCreateGlobalVector(user.da, &Unew);
    SetInitialConditions(U, &user);
    ApplyBoundaryConditions(U, &user);
    
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
        
        // Print centerline values for comparison
        PetscScalar **u;
        DMDAVecGetArray(user.da, U, &u);
        PetscPrintf(PETSC_COMM_SELF, "\nCenterline values (x=0.5):\n");
        for (PetscInt j = 0; j < user.ny; j++) {
            PetscInt i = user.nx/2;
            PetscPrintf(PETSC_COMM_SELF, "y=%.3f, T=%.6f\n", 
                       j*user.dy, u[j][i]);
        }
        DMDAVecRestoreArray(user.da, U, &u);
    }
    
    // Output solution
    PetscViewer viewer;
    PetscViewerASCIIOpen(PETSC_COMM_WORLD, "solution.csv", &viewer);
    PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_CSV);
    
    PetscScalar **u;
    DMDAVecGetArray(user.da, U, &u);
    
    if (rank == 0) {
        PetscViewerASCIIPrintf(viewer, "x,y,T\n");
    }
    
    PetscInt i, j, xs, ys, xn, yn;
    DMDAGetCorners(user.da, &xs, &ys, NULL, &xn, &yn, NULL);
    
    for (j = ys; j < ys+yn; j++) {
        PetscReal y = j * user.dy;
        for (i = xs; i < xs+xn; i++) {
            PetscReal x = i * user.dx;
            PetscViewerASCIIPrintf(viewer, "%f,%f,%f\n", x, y, u[j][i]);
        }
    }
    
    DMDAVecRestoreArray(user.da, U, &u);
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

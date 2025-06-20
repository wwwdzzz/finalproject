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
    PetscInt  current_step; // Current time step index
    DM        da;        // Distributed array context
    Mat       A;         // Matrix for implicit method
    KSP       ksp;       // Linear solver context
    PetscReal bc_values[4]; // Boundary condition values
    PetscInt  bc_types[4];  // Boundary condition types
    PetscBool verification; // Verification mode flag
} AppCtx;

// Exact solution for verification
PetscReal ExactSolution(PetscReal x, PetscReal y, PetscReal t, AppCtx* user) {
    return (1.0 - x) + exp(-t) * sin(M_PI*x) * cos(M_PI*y);
}

// Forcing term derived from exact solution
PetscReal ForcingTerm(PetscReal x, PetscReal y, PetscReal t, AppCtx* user) {
    PetscReal term1 = -exp(-t) * sin(M_PI*x) * cos(M_PI*y);  // du/dt term
    PetscReal term2 = user->kappa * (M_PI*M_PI) * exp(-t) * sin(M_PI*x) * cos(M_PI*y);  // diffusion term
    return user->rho * user->c * term1 + term2;
}

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
                    MatSetValue(user->A, row, row, 1.0, INSERT_VALUES);
                    continue;
                }
                
                // Right boundary (Dirichlet)
                if (i == user->nx-1) {
                    MatSetValue(user->A, row, row, 1.0, INSERT_VALUES);
                    continue;
                }
                
                // Bottom boundary (Neumann: ∂u/∂y = 0)
                if (j == 0) {
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
                    values[ncols] = -2.0*coeff_y;
                    ncols++;
                    MatSetValues(user->A, 1, &row, ncols, col, values, INSERT_VALUES);
                    continue;
                }
                
                // Top boundary (Neumann: ∂u/∂y = 0)
                if (j == user->ny-1) {
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
                    values[ncols] = -2.0*coeff_y;
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
            if (user->verification) {
                u[j][i] = ExactSolution(x, y, 0.0, user);
            } else {
                u[j][i] = (1.0 - x) + 0.1 * sin(2*M_PI*x) * sin(2*M_PI*y);
            }
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
            if (user->verification) {
                PetscReal x = i * user->dx;
                PetscReal y = j * user->dy;
                PetscReal t = user->dt * (PetscReal)user->current_step;
                unew[j][i] = u[j][i] + 
                    coeff_x * (u[j][i-1] - 2.0*u[j][i] + u[j][i+1]) +
                    coeff_y * (u[j-1][i] - 2.0*u[j][i] + u[j+1][i]) +
                    user->dt * ForcingTerm(x, y, t, user) / (user->rho * user->c);
            } else {
                unew[j][i] = u[j][i] + 
                    coeff_x * (u[j][i-1] - 2.0*u[j][i] + u[j][i+1]) +
                    coeff_y * (u[j-1][i] - 2.0*u[j][i] + u[j+1][i]);
            }
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
    
    PetscInt i, j, xs, ys, xn, yn;
    DMDAGetCorners(user->da, &xs, &ys, NULL, &xn, &yn, NULL);
    
    // Left boundary (Dirichlet)
    if (xs == 0) {
        for (j = ys; j < ys+yn; j++) {
            b_array[j][0] = user->bc_values[0];
        }
    }
    
    // Right boundary (Dirichlet)
    if (xs + xn == user->nx) {
        for (j = ys; j < ys+yn; j++) {
            b_array[j][user->nx-1] = user->bc_values[1];
        }
    }
    
    // Add forcing term for verification
    if (user->verification) {
        for (j = ys; j < ys+yn; j++) {
            PetscReal y = j * user->dy;
            for (i = xs; i < xs+xn; i++) {
                PetscReal x = i * user->dx;
                PetscReal t = user->dt * (PetscReal)user->current_step;
                b_array[j][i] += user->dt * ForcingTerm(x, y, t, user) / (user->rho * user->c);
            }
        }
    }
    
    DMDAVecRestoreArray(user->da, b, &b_array);
    
    // Solve the linear system: A*Unew = b
    KSPSolve(user->ksp, b, Unew);
    
    // Ensure boundary conditions are satisfied (important for parallel runs)
    ApplyBoundaryConditions(Unew, user);
    
    VecDestroy(&b);
    PetscFunctionReturn(0);
}

PetscErrorCode CalculateError(Vec U, PetscReal t, AppCtx* user, PetscReal* max_error) {
    PetscFunctionBeginUser;
    PetscScalar **u;
    DMDAVecGetArray(user->da, U, &u);
    
    PetscInt i, j, xs, ys, xn, yn;
    DMDAGetCorners(user->da, &xs, &ys, NULL, &xn, &yn, NULL);
    
    PetscReal error = 0.0;
    for (j = ys; j < ys+yn; j++) {
        PetscReal y = j * user->dy;
        for (i = xs; i < xs+xn; i++) {
            PetscReal x = i * user->dx;
            PetscReal exact = ExactSolution(x, y, t, user);
            PetscReal current_error = PetscAbsReal(exact - u[j][i]);
            if (current_error > error) error = current_error;
        }
    }
    
    DMDAVecRestoreArray(user->da, U, &u);
    
    // Get global maximum error across all processors
    MPI_Allreduce(&error, max_error, 1, MPIU_REAL, MPI_MAX, PETSC_COMM_WORLD);
    
    PetscFunctionReturn(0);
}

PetscErrorCode RunConvergenceTest(AppCtx* user) {
    PetscFunctionBeginUser;
    PetscInt num_refinements = 5;
    PetscReal spatial_errors[num_refinements];
    PetscReal temporal_errors[num_refinements];
    PetscReal dx_values[num_refinements];
    PetscReal dt_values[num_refinements];
    
    PetscMPIInt rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    
    // Spatial convergence test (fix very small dt)
    if (rank == 0) {
        PetscPrintf(PETSC_COMM_SELF, "\n=== Spatial Convergence Test ===\n");
        PetscPrintf(PETSC_COMM_SELF, "dx        Error       Rate\n");
    }
    
    PetscReal fixed_dt = 1e-6;  // Very small to eliminate temporal error
    
    for (PetscInt ref = 0; ref < num_refinements; ref++) {
        PetscInt nx = 10 * (1 << ref);  // 10, 20, 40, 80, 160
        user->nx = nx;
        user->ny = nx;
        user->dx = 1.0 / (user->nx - 1);
        user->dy = user->dx;
        user->dt = fixed_dt;
        
        // Reinitialize DA and matrix
        if (user->da) DMDestroy(&user->da);
        if (user->method == 1 && user->A) MatDestroy(&user->A);
        if (user->method == 1 && user->ksp) KSPDestroy(&user->ksp);
        
        DMDACreate2d(PETSC_COMM_WORLD, 
                    DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED,
                    DMDA_STENCIL_STAR,
                    user->nx, user->ny,
                    PETSC_DECIDE, PETSC_DECIDE,
                    1, 1,
                    NULL, NULL,
                    &user->da);
        DMSetFromOptions(user->da);
        DMSetUp(user->da);
        
        Initialize(user);
        
        // Run simulation
        Vec U, Unew;
        DMCreateGlobalVector(user->da, &U);
        DMCreateGlobalVector(user->da, &Unew);
        SetInitialConditions(U, user);
        ApplyBoundaryConditions(U, user);
        
        PetscInt nsteps = (PetscInt)(user->T_final / user->dt) + 1;
        for (user->current_step = 0; user->current_step < nsteps; user->current_step++) {
            if (user->method == 0) ExplicitEulerStep(U, Unew, user);
            else ImplicitEulerStep(U, Unew, user);
            VecSwap(U, Unew);
        }
        
        // Calculate error
        PetscReal max_error;
        CalculateError(U, user->T_final, user, &max_error);
        
        dx_values[ref] = user->dx;
        spatial_errors[ref] = max_error;
        
        // Print results
        if (rank == 0) {
            if (ref == 0) {
                PetscPrintf(PETSC_COMM_SELF, "%.4e  %.4e  -\n", 
                           user->dx, max_error);
            } else {
                PetscReal rate = log(spatial_errors[ref-1]/max_error) / log(2.0);
                PetscPrintf(PETSC_COMM_SELF, "%.4e  %.4e  %.2f\n", 
                           user->dx, max_error, rate);
            }
        }
        
        VecDestroy(&U);
        VecDestroy(&Unew);
    }
    
    // Temporal convergence test (fix fine spatial grid)
    if (rank == 0) {
        PetscPrintf(PETSC_COMM_SELF, "\n=== Temporal Convergence Test ===\n");
        PetscPrintf(PETSC_COMM_SELF, "dt        Error       Rate\n");
    }
    
    user->nx = 100;  // Fixed fine spatial grid
    user->ny = 100;
    user->dx = 1.0 / (user->nx - 1);
    user->dy = user->dx;
    
    // Reinitialize DA and matrix
    if (user->da) DMDestroy(&user->da);
    if (user->method == 1 && user->A) MatDestroy(&user->A);
    if (user->method == 1 && user->ksp) KSPDestroy(&user->ksp);
    
    DMDACreate2d(PETSC_COMM_WORLD, 
                DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED,
                DMDA_STENCIL_STAR,
                user->nx, user->ny,
                PETSC_DECIDE, PETSC_DECIDE,
                1, 1,
                NULL, NULL,
                &user->da);
    DMSetFromOptions(user->da);
    DMSetUp(user->da);
    
    Initialize(user);
    
    for (PetscInt ref = 0; ref < num_refinements; ref++) {
        PetscReal dt = 0.01 / (1 << (2*ref));  // 0.01, 0.0025, 0.000625, etc.
        user->dt = dt;
        
        // Run simulation
        Vec U, Unew;
        DMCreateGlobalVector(user->da, &U);
        DMCreateGlobalVector(user->da, &Unew);
        SetInitialConditions(U, user);
        ApplyBoundaryConditions(U, user);
        
        PetscInt nsteps = (PetscInt)(user->T_final / user->dt) + 1;
        for (user->current_step = 0; user->current_step < nsteps; user->current_step++) {
            if (user->method == 0) ExplicitEulerStep(U, Unew, user);
            else ImplicitEulerStep(U, Unew, user);
            VecSwap(U, Unew);
        }
        
        // Calculate error
        PetscReal max_error;
        CalculateError(U, user->T_final, user, &max_error);
        
        dt_values[ref] = user->dt;
        temporal_errors[ref] = max_error;
        
        // Print results
        if (rank == 0) {
            if (ref == 0) {
                PetscPrintf(PETSC_COMM_SELF, "%.4e  %.4e  -\n", 
                           user->dt, max_error);
            } else {
                PetscReal rate = log(temporal_errors[ref-1]/max_error) / log(2.0);
                PetscPrintf(PETSC_COMM_SELF, "%.4e  %.4e  %.2f\n", 
                           user->dt, max_error, rate);
            }
        }
        
        VecDestroy(&U);
        VecDestroy(&Unew);
    }
    
    PetscFunctionReturn(0);
}

int main(int argc, char **argv) {
    AppCtx         user;
    Vec            U, Unew;
    PetscInt       nsteps;
    PetscMPIInt    rank;
    PetscLogDouble t1, t2;
    PetscBool      verification = PETSC_FALSE;
    
    PetscInitialize(&argc, &argv, NULL, NULL);
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    
    // Initialize user context
    PetscMemzero(&user, sizeof(AppCtx));
    user.current_step = 0;
    
    // Default parameters
    user.kappa = 1.0;
    user.rho = 1.0;
    user.c = 1.0;
    user.method = 1;      // Default to implicit
    user.nx = 20;
    user.ny = 20;
    user.dt = 0.0001;
    user.T_final = 0.01;
    user.verification = PETSC_FALSE;
    
    // Boundary conditions
    user.bc_values[0] = 1.0;  // Left
    user.bc_values[1] = 0.0;  // Right
    user.bc_values[2] = 0.0;  // Bottom (unused)
    user.bc_values[3] = 0.0;  // Top (unused)
    
    user.bc_types[0] = 0;     // Left: Dirichlet
    user.bc_types[1] = 0;     // Right: Dirichlet
    user.bc_types[2] = 1;     // Bottom: Neumann
    user.bc_types[3] = 1;     // Top: Neumann
    
    // Process command line options
    PetscOptionsGetInt(NULL, NULL, "-nx", &user.nx, NULL);
    PetscOptionsGetInt(NULL, NULL, "-ny", &user.ny, NULL);
    PetscOptionsGetReal(NULL, NULL, "-dt", &user.dt, NULL);
    PetscOptionsGetReal(NULL, NULL, "-T", &user.T_final, NULL);
    PetscOptionsGetInt(NULL, NULL, "-method", &user.method, NULL);
    PetscOptionsGetBool(NULL, NULL, "-verify", &verification, NULL);
    
    user.dx = 1.0 / (user.nx - 1);
    user.dy = 1.0 / (user.ny - 1);
    nsteps = (PetscInt)(user.T_final / user.dt) + 1;
    user.verification = verification;
    
    if (verification) {
        RunConvergenceTest(&user);
    } else {
        // Regular simulation mode
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
        for (user.current_step = 0; user.current_step < nsteps; user.current_step++) {
            if (user.method == 0) {
                ExplicitEulerStep(U, Unew, &user);
            } else {
                ImplicitEulerStep(U, Unew, &user);
            }
            VecSwap(U, Unew);
            
            if (user.current_step % 100 == 0 && rank == 0) {
                PetscPrintf(PETSC_COMM_SELF, "Step %d/%d, Time %.4f/%.4f\n", 
                           user.current_step, nsteps, user.current_step*user.dt, user.T_final);
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
    }
    
    // Clean up
    if (user.method == 1) {
        MatDestroy(&user.A);
        KSPDestroy(&user.ksp);
    }
    DMDestroy(&user.da);
    PetscFinalize();
    return 0;
}

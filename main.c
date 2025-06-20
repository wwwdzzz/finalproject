#include <petscdm.h>
#include <petscdmda.h>
#include <petscmat.h>
#include <petscvec.h>
#include <petscksp.h>
#include <math.h>
#include <time.h>
#include <stdio.h>

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

PetscErrorCode WriteSolutionToFileSerial(Vec U, AppCtx* user) {
    PetscFunctionBeginUser;
    PetscScalar *u;
    PetscInt i, xs, xn;
    FILE *fp;

    VecGetArray(U, &u);
    DMDAGetCorners(user->da, &xs, NULL, NULL, &xn, NULL, NULL);

    // Only rank 0 writes the file (gathers all data)
    if (user->da->comm->rank == 0) {
        fp = fopen("solution_serial.txt", "w");
        if (!fp) {
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Cannot open solution_serial.txt");
        }
        fprintf(fp, "X\tY\tTemperature\n");  // Header line

        // Write rank 0's portion
        for (i = xs; i < xs+xn; i++) {
            PetscReal x = i * user->dx;
            fprintf(fp, "%.6f\t0.0\t%.6f\n", x, u[i]);
        }

        // Receive and write data from other ranks
        PetscMPIInt size;
        MPI_Comm_size(PETSC_COMM_WORLD, &size);
        
        if (size > 1) {
            PetscInt *counts = (PetscInt*)malloc(size * sizeof(PetscInt));
            PetscInt *displs = (PetscInt*)malloc(size * sizeof(PetscInt));
            
            // Gather all counts
            PetscInt my_count = xn;
            MPI_Gather(&my_count, 1, MPI_INT, counts, 1, MPI_INT, 0, PETSC_COMM_WORLD);
            
            // Calculate displacements
            displs[0] = 0;
            for (PetscMPIInt r = 1; r < size; r++) {
                displs[r] = displs[r-1] + counts[r-1];
            }
            
            // Allocate receive buffer
            PetscScalar *recv_buf = (PetscScalar*)malloc(user->nx * sizeof(PetscScalar));
            
            // Gather all data
            MPI_Gatherv(u, xn, MPIU_SCALAR, recv_buf, counts, displs, MPIU_SCALAR, 0, PETSC_COMM_WORLD);
            
            // Write other ranks' data
            for (PetscMPIInt r = 1; r < size; r++) {
                for (i = displs[r]; i < displs[r] + counts[r]; i++) {
                    PetscReal x = i * user->dx;
                    fprintf(fp, "%.6f\t0.0\t%.6f\n", x, recv_buf[i]);
                }
            }
            
            free(counts);
            free(displs);
            free(recv_buf);
        }
        
        fclose(fp);
    } else {
        // Other ranks send their data to rank 0
        PetscInt my_count = xn;
        MPI_Gather(&my_count, 1, MPI_INT, NULL, 1, MPI_INT, 0, PETSC_COMM_WORLD);
        MPI_Gatherv(u, xn, MPIU_SCALAR, NULL, NULL, NULL, MPIU_SCALAR, 0, PETSC_COMM_WORLD);
    }

    VecRestoreArray(U, &u);
    PetscFunctionReturn(0);
}

PetscErrorCode WriteSolutionToFileParallel(Vec U, AppCtx* user) {
    PetscFunctionBeginUser;
    PetscScalar *u;
    PetscInt i, xs, xn;
    MPI_File fh;
    char *buffer;
    size_t buffer_size;
    PetscMPIInt rank;
    MPI_Status status;

    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    VecGetArray(U, &u);
    DMDAGetCorners(user->da, &xs, NULL, NULL, &xn, NULL, NULL);

    // First, have rank 0 write the header
    if (rank == 0) {
        char header[] = "X\tY\tTemperature\n";
        MPI_File_open(PETSC_COMM_SELF, "solution_parallel.txt", 
                     MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
        MPI_File_write(fh, header, strlen(header), MPI_CHAR, &status);
        MPI_File_close(&fh);
    }
    MPI_Barrier(PETSC_COMM_WORLD);

    // Now each process writes its portion
    // Calculate required buffer size (3 values per point, each up to 20 chars + tabs/newline)
    buffer_size = xn * 64;  // Generous estimate
    buffer = (char*)malloc(buffer_size);
    char *ptr = buffer;

    for (i = xs; i < xs+xn; i++) {
        PetscReal x = i * user->dx;
        ptr += sprintf(ptr, "%.6f\t0.0\t%.6f\n", x, u[i]);
    }

    // Open file in append mode
    MPI_File_open(PETSC_COMM_WORLD, "solution_parallel.txt", 
                 MPI_MODE_WRONLY | MPI_MODE_APPEND, MPI_INFO_NULL, &fh);
    
    // Each process writes its portion
    MPI_File_write_shared(fh, buffer, ptr - buffer, MPI_CHAR, &status);
    
    MPI_File_close(&fh);
    free(buffer);
    VecRestoreArray(U, &u);

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
    user.method = 1;  // Default to implicit
    user.nx = 100;
    user.dt = 0.001;
    user.T_final = 0.1;
    
    // Read command line options
    PetscOptionsGetInt(NULL, NULL, "-nx", &user.nx, NULL);
    PetscOptionsGetReal(NULL, NULL, "-dt", &user.dt, NULL);
    PetscOptionsGetReal(NULL, NULL, "-T", &user.T_final, NULL);
    PetscOptionsGetInt(NULL, NULL, "-method", &user.method, NULL);
    
    user.dx = 1.0 / (user.nx - 1);
    nsteps = (PetscInt)(user.T_final / user.dt) + 1;
    
    // Create distributed array
    DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_GHOSTED, user.nx, 1, 1, NULL, &user.da);
    DMSetFromOptions(user.da);
    DMSetUp(user.da);
    
    // Initialize solver
    Initialize(&user);
    
    // Create vectors
    DMCreateGlobalVector(user.da, &U);
    DMCreateGlobalVector(user.da, &Unew);
    SetInitialConditions(U, &user);
    
    // Time stepping
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
    
    // Output solution
    WriteSolutionToFileSerial(U, &user);    // Serial version (gathers to rank 0)
    WriteSolutionToFileParallel(U, &user);  // Parallel version (MPI I/O)
    
    // Clean up
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

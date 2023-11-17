CS 179: GPU Computing
Lab 1: Introduction to CUDA
Name:

================================================================================
Question 1: Common Errors (20 points)
================================================================================

--------------------------------------------------------------------------------
1.1
--------------------------------------------------------------------------------
Issue: int *a = 3; we cannot assign an integer value to a "pointer" which actually points to a block of memory

Fix: int* a = (int*)malloc(sizeof(int));

--------------------------------------------------------------------------------
1.2
--------------------------------------------------------------------------------
Issue: int *a, b; forget that b is a pointer instead of an integer

Fix: int *a, *b;

--------------------------------------------------------------------------------
1.3
--------------------------------------------------------------------------------
Issue: int i, *a = (int *) malloc(1000); we should allocate 1000*sizeof(int), that is 4000 bytes of memory, instead of 1000

Fix: int i, *a = (int *) malloc(1000*sizeof(int));

--------------------------------------------------------------------------------
1.4
--------------------------------------------------------------------------------
Issue: only assign memory for the first dimension, we should also assign memory for the second dimension (the 100 integers)

Fix:
    for(int i=0; i<3; ++i){
        a[i] = (int*)malloc(100 * sizeof(int));
    }
--------------------------------------------------------------------------------
1.5
--------------------------------------------------------------------------------
Issue: since a is a pointer that is already allocated to an address, so if(!a) will always be evaluated to false, but our logic is to judge whether the user input is 0

Fix: if(*a==0)

================================================================================
Question 2: Parallelization (30 points)
================================================================================

--------------------------------------------------------------------------------
2.1
--------------------------------------------------------------------------------
y_1[n] = x[n - 1] + x[n] + x[n + 1] will have a faster implementation, because the calculation of y_1[i] is independent
y_2[n] = y_2[n - 2] + y_2[n - 1] + x[n] will have a faster implementation because the calculation of y_2[i] is dependent on the result of previous y_2

--------------------------------------------------------------------------------
2.2
--------------------------------------------------------------------------------
y[n] = c * x[n] + (1 - c) * y[n - 1]
     = c * x[n] + (1 - c) * (c * x[n - 1]  + (1-c) * y[n-2])
     = c * x[n] + (1 - c) * c * x [n - 1] + (1-c)^2 * y[n-2]
since c is almost 1, so we can omit (1-c)^2 * y[n-2], which gives us c * x[n] + (1 - c) * c * x [n - 1]

================================================================================
Question 3: Small-Kernel Convolution (50 points)
================================================================================

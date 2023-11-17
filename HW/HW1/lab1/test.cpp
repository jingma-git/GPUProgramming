#include <stdio.h>
#include <stdlib.h>

void test1(){
    // int *a = 3;
    // *a = *a + 2;
    // printf("%d\n", *a);

    int* a = (int*)malloc(sizeof(int));
    *a = 3;
    *a = *a + 2;
    printf("%d\n", *a);
    free(a);
}


void test2() {
    // int *a, b;
    // a = (int *) malloc(sizeof (int));
    // b = (int *) malloc(sizeof (int));

    // if (!(a && b)) {
    //     printf("Out of memory\n");
    //     exit(-1);
    // }
    // *a = 2;
    // *b = 3;

    int *a, *b;
    a = (int *) malloc(sizeof (int));
    b = (int *) malloc(sizeof (int));

    if (!(a && b)) {
        printf("Out of memory\n");
        exit(-1);
    }
    *a = 2;
    *b = 3;
    free(a);
    free(b);
}

void test3() {
    // int i, *a = (int *) malloc(1000);

    // if (!a) {
    //     printf("Out of memory\n");
    //     exit(-1);
    // }
    // for (i = 0; i < 1000; i++){
    //     *(i + a) = i;
    //     printf("%d\n", a[i]);
    // }

    int i, *a = (int *) malloc(1000*sizeof(int));

    if (!a) {
        printf("Out of memory\n");
        exit(-1);
    }
    for (i = 0; i < 1000; i++){
        *(i + a) = i;
        printf("%d\n", a[i]);
    }
    free(a);
}


void test4() {
    // int **a = (int **) malloc(3 * sizeof (int *));
    // a[1][1] = 5;

    int **a = (int **) malloc(3 * sizeof (int *));
    for(int i=0; i<3; ++i){
        a[i] = (int*)malloc(100 * sizeof(int));
    }
    a[1][1] = 5;
    printf("%d, %d\n", a[1][1], a[2][99]);
    for(int i=0; i<3; ++i){
        free(a[i]);
    }
}

void test5() {
    // int *a = (int *) malloc(sizeof (int));
    // scanf("%d", a);
    // if (!a)
    //     printf("Value is 0\n");

    int *a = (int *) malloc(sizeof (int));
    scanf("%d", a);
    if (*a == 0)
        printf("Value is 0\n");
}

int main(){
    // test1();
    // test2();
    // test3();
    // test4();
    test5();
    return 0;
}
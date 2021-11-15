#include "hmm.h"
#include <math.h>

int main(int argc, char *argv[])
{
    printf("We have %d arguments:\n", argc);
    for (int i = 0; i < argc; ++i) {
        printf("[%d] %s\n", i, argv[i]);
    }
    return 0;


/*
    HMM hmms[5];
    load_models( "modellist.txt", hmms, 5);
    dump_models( hmms, 5);
*/
    // HMM hmm_initial;
    // loadHMM( &hmm_initial, "../model_init.txt" );
    // dumpHMM( stderr, &hmm_initial );

    // printf("log(0.5) = %f\n", log(1.5) );
    // return 0;
}

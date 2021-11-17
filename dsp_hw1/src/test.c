#include "hmm.h"
#include <math.h>

#define SEQ_COUNT  10000
#define SEQ_T  50
#define MODEL_COUNT 5

char sequenceData[SEQ_COUNT][SEQ_T];

/* Read the train_seq_0x.txt to a 2D char array */
static void loadSequence(const char *filename)
{
    FILE *fp = open_or_die(filename, "r");
    char token[50] = "";
    int line_count = 0;

    while (fscanf(fp, "%[^\n] ", token) > 0) {
        for(int alphabet = 0; alphabet < 50; alphabet++){
            sequenceData[line_count][alphabet] = token[alphabet];
        }
        line_count++;
    }
    return;
}

int main(int argc, char *argv[])
{
    // Error in command line
    if (argc != 4){
        printf("The number of arguments is wrong.\n");
        return 0;
    }

    // Load test seq
    loadSequence(argv[2]);

    // Clear file content
    FILE *result_fp = open_or_die( argv[3], "w");
    fclose(result_fp);

    // Load HMM models
    HMM modelList[MODEL_COUNT];
	load_models(argv[1], modelList, MODEL_COUNT);

    int stateNum = modelList[0].state_num;
    double delta[stateNum][SEQ_T];
    double highestProb = -1000.0;
    int resultModelIndex = 0;

    int currentSequenceLine = 0;


    while(currentSequenceLine < 2500){
        highestProb = -1000.0;
        resultModelIndex = 0;

        // Initialize the array
        for( int i = 0 ; i < stateNum ; i++ ){
            for(int j = 0 ; j < SEQ_T ; j++ ){
                delta[i][j] = 0.0;
            }
        }

        // δ(Delta): Go through 5 models
        for(int modelIndex = 0; modelIndex < MODEL_COUNT ; modelIndex++){
            // Initialization
            for (int i = 0 ; i < stateNum ; i++){
                int tempO = (int)sequenceData[currentSequenceLine][0] - 65; // o(1)
                delta[i][0] = modelList[modelIndex].initial[i] * modelList[modelIndex].observation[tempO][i];
            }

            // Recursion
            for(int t = 1 ; t < SEQ_T ; t++){
                for(int j = 0 ; j < stateNum ; j++){
                    double maxState = 0.0;
                    for (int i = 0 ; i < stateNum ; i++){
                        double tempSum = delta[i][t-1] * modelList[modelIndex].transition[i][j];
                        if(maxState < tempSum)
                            maxState = tempSum;
                    }
                    int tempO = (int)sequenceData[currentSequenceLine][t] - 65; // o(t)
                    delta[j][t] = maxState * modelList[modelIndex].observation[tempO][j];
                }
            }

            // Find the max δT(i)
            double temp_highProb = -1000.0;
            for (int ite_i = 0 ; ite_i < stateNum ; ite_i++){
                if(temp_highProb < delta[ite_i][SEQ_T-1]){
                    temp_highProb = delta[ite_i][SEQ_T-1];
                }
            }

            if(highestProb < temp_highProb){
                highestProb = temp_highProb;
                resultModelIndex = modelIndex+1;
            }

        }

        FILE *result_fp = open_or_die( argv[3], "a");
        fprintf( result_fp, "model_0%d.txt %12.5e\n", resultModelIndex, highestProb);
        fclose(result_fp);

        currentSequenceLine++;
    }
    

    return 0;
}

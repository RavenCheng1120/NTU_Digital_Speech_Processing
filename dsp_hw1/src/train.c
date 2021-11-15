#include "hmm.h"
#include <math.h>

#define SEQ_COUNT  10000
#define SEQ_T  50
/* 
A - transition: 6*6
B - observation: 6*6, column 1 means state 1 and row means the alphabet
π - initial: 1*6

α - N*T = 6*50
β - N*T = 6*50
γ - N*T = 6*50
ϵ - (T−1)*N*N = 49*6*6

Alphabet to int: 
    A, B, C, D, E, F = 65, 66, 67, 68, 69, 70
    Need to subtract 65 to use in observation array
*/

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
    if (argc != 5){
        printf("The number of arguments is wrong.\n");
        return 0;
    }

    loadSequence("./data/train_seq_01.txt");
    
    // Load initial model
    HMM hmm_initial;
    loadHMM(&hmm_initial, argv[2]);

    if(hmm_initial.state_num <= 0){
        printf("loadHMM went wrong.\n");
        return 0;
    }

    /************** 
    * Train model *
    ***************/
    float alpha[hmm_initial.state_num][SEQ_T];
    float beta[hmm_initial.state_num][SEQ_T];
    float gamma[hmm_initial.state_num][SEQ_T];
    float epsilon[SEQ_T-1][hmm_initial.state_num][hmm_initial.state_num];

    int currentSequenceLine = 0;

    while(currentSequenceLine < 10000){
        // Initialize the array
        for( int i = 0 ; i < hmm_initial.state_num ; i++ ){
            for(int j = 0 ; j < SEQ_T ; j++ ){
                alpha[i][j] = 0.0;
                beta[i][j] = 0.0;
                gamma[i][j] = 0.0;
                for(int k = 0 ; k < hmm_initial.state_num ; k++)
                    epsilon[j][i][k] = 0.0;
            }
        }

        // --- Forward algorithm ---
        // 1. Initialization
        for(int i = 0 ; i < hmm_initial.state_num ; i++){
            int tempO = (int)sequenceData[currentSequenceLine][0] - 65; // o(1)
            alpha[i][0] = hmm_initial.initial[i] * hmm_initial.observation[tempO][i];
            // printf("tempO=%d\talpha=%f\n",tempO,alpha[i][0]);
        }
        // 2. Induction & Termination
        for(int t = 0 ; t < SEQ_T-1 ; t++ ){
            for( int j = 0 ; j < hmm_initial.state_num ; j++ ){
                float previousAlpha = 0; 
                for(int i = 0 ; i < hmm_initial.state_num ; i++)
                    previousAlpha += (alpha[i][t] * hmm_initial.transition[i][j]); // previous alphas
                int tempO = (int)sequenceData[currentSequenceLine][t+1] - 65; // o(t+1)
                alpha[j][t+1] = previousAlpha*hmm_initial.observation[tempO][j];
            }
        }

        // --- Backward algorithm ---
        // 1. Initializatoin
        for(int i = 0 ; i < hmm_initial.state_num ; i++){
            beta[i][SEQ_T-1] = 1;
        }

        // 2. Induction
        for(int t = SEQ_T-2 ; t >= 0 ; t-- ){
            for( int i = 0 ; i < hmm_initial.state_num ; i++ ){
                for(int j = 0 ; j < hmm_initial.state_num ; j++){
                    int tempO = (int)sequenceData[currentSequenceLine][t+1] - 65; // o(t+1)
                    beta[i][t] += (hmm_initial.observation[tempO][j] * hmm_initial.transition[i][j] * beta[j][t+1]);
                }
            }
        }

        // --- γ: Temporary variable ---
        for(int t = 0 ; t < SEQ_T ; t++ ){
            for( int i = 0 ; i < hmm_initial.state_num ; i++ ){
                float tempSum = 0.0;
                for(int j = 0 ; j < hmm_initial.state_num ; j++)
                    tempSum = alpha[j][t] * beta[j][t];
                gamma[i][t] = (alpha[i][t] * beta[i][t])  / tempSum;
            }
        }

        // --- ϵ: The probability of transition from state i to state j given observation and model ---
        for(int t = 0 ; t < SEQ_T ; t++ ){
            for( int i = 0 ; i < hmm_initial.state_num ; i++ ){
                for(int j = 0 ; j < hmm_initial.state_num ; j++){
                    int tempO = (int)sequenceData[currentSequenceLine][t+1] - 65;
                    float denominator = 0.0;

                    // Denominator
                    for(int inner_i = 0 ; inner_i < hmm_initial.state_num ; inner_i++){
                        for(int inner_j = 0 ; inner_j < hmm_initial.state_num ; inner_j++){
                            denominator = alpha[inner_i][t] * hmm_initial.transition[inner_i][inner_j] * hmm_initial.observation[tempO][inner_j] * beta[inner_j][t+1];
                        }
                    }

                    // Epsilon
                    epsilon[t][i][j] = (alpha[i][t] * hmm_initial.transition[i][j] * hmm_initial.observation[tempO][j] * beta[j][t+1]) / denominator;
                }
            }
        }

        // --- Update parameters ---
        // 1. π
        for(int i = 0 ; i < hmm_initial.state_num ; i++)
            hmm_initial.initial[i] = gamma[i][1];

        // 2. A
        for(int i = 0 ; i < hmm_initial.state_num ; i++ ){
            for(int j = 0 ; j < hmm_initial.state_num ; j++){
                float epsilon_sum = 0.0;
                float gamma_sum = 0.0;
                for(int t = 0; t < SEQ_T-1 ; t++){
                    epsilon_sum += epsilon[t][i][j];
                    gamma_sum += gamma[i][t];
                }
                hmm_initial.transition[i][j] = epsilon_sum / gamma_sum;
            }
        }

        // 3. B
        for(int i = 0 ; i < hmm_initial.state_num ; i++ ){
            for(int k = 0 ; k < hmm_initial.observ_num ; k++){
                float gamma_sum_t = 0.0;
                float gamma_sum_observ = 0.0;
                for(int t = 0; t < SEQ_T ; t++){
                    gamma_sum_t += gamma[i][t];
                    if (k+65 == (int)sequenceData[currentSequenceLine][t])
                        gamma_sum_observ += gamma[i][t];
                }
                hmm_initial.observation[k][i] = gamma_sum_observ / gamma_sum_t;
            }
        }

        currentSequenceLine++;
    }

    

    // Dump trained model
    FILE *dump_fp = open_or_die( argv[4], "w+");
    dumpHMM( dump_fp, &hmm_initial );

    // for( int i = 0 ; i < hmm_initial.observ_num ; i++ ){
    //     for(int j = 0 ; j < hmm_initial.state_num ; j++ )
    //         printf("%f ",hmm_initial.observation[i][j]);
    //     printf("\n");
    // }
    return 0;
}

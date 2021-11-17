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

    loadSequence(argv[3]);
    
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
    double alpha[hmm_initial.state_num][SEQ_T];
    double beta[hmm_initial.state_num][SEQ_T];
    double gammaParam[hmm_initial.state_num][SEQ_T];
    double epsilon[SEQ_T-1][hmm_initial.state_num][hmm_initial.state_num];
    HMM new_hmm;
    loadHMM(&new_hmm, argv[2]);
    

    double a_denominator[hmm_initial.state_num][hmm_initial.state_num];
    double a_molecular[hmm_initial.state_num][hmm_initial.state_num];
    double b_denominator[hmm_initial.state_num][hmm_initial.state_num];
    double b_molecular[hmm_initial.state_num][hmm_initial.state_num];


    for(int iteration = 0; iteration < atoi(argv[1]); iteration++){
        // Initialize the array
        for( int i = 0 ; i < hmm_initial.state_num ; i++ ){
            for(int j = 0 ; j < hmm_initial.state_num ; j++ ){
                a_denominator[i][j] = 0.0;
                a_molecular[i][j] = 0.0;
                b_denominator[i][j] = 0.0;
                b_molecular[i][j] = 0.0;
            }
        }

        int currentSequenceLine = 0;

        while(currentSequenceLine < 1000){
            // Initialize the array
            for( int i = 0 ; i < hmm_initial.state_num ; i++ ){
                for(int j = 0 ; j < SEQ_T ; j++ ){
                    alpha[i][j] = 0.0;
                    beta[i][j] = 0.0;
                    gammaParam[i][j] = 0.0;
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
                    double previousAlpha = 0.0; 
                    for(int i = 0 ; i < hmm_initial.state_num ; i++){
                        previousAlpha += (alpha[i][t] * hmm_initial.transition[i][j]); // previous alphas
                    }
                    // Debug
                    // printf("alpha[i][t]:%.7f A:%.2f previousAlpha:%.15f\n",alpha[0][t], hmm_initial.transition[0][j], previousAlpha);
                    int tempO = (int)sequenceData[currentSequenceLine][t+1] - 65; // o(t+1)
                    alpha[j][t+1] = previousAlpha * hmm_initial.observation[tempO][j];
                }
                // printf("\n");
            }

            // Debug
            // float temp = 0.0;
            // for( int i = 0 ; i < hmm_initial.state_num ; i++ ){
            //     temp += alpha[i][SEQ_T-1];
            // }
            // printf("%.15f\n", temp);


            // --- Backward algorithm ---
            // 1. Initializatoin
            for(int i = 0 ; i < hmm_initial.state_num ; i++){
                beta[i][SEQ_T-1] = 1;
            }

            // 2. Induction
            for(int t = SEQ_T-2 ; t >= 0 ; t-- ){
                int tempO = (int)sequenceData[currentSequenceLine][t+1] - 65; // o(t+1)
                for(int i = 0 ; i < hmm_initial.state_num ; i++ ){
                    for(int j = 0 ; j < hmm_initial.state_num ; j++){
                        beta[i][t] += (hmm_initial.observation[tempO][j] * hmm_initial.transition[i][j] * beta[j][t+1]);
                    }
                }
            }

            // for(int i = 0 ; i < 6 ; i++)
            //     printf("beta=%f\n", beta[i][0]);

            // --- γ: Temporary variable ---
            for(int t = 0 ; t < SEQ_T ; t++ ){
                double tempSum = 0.0;
                for(int j = 0 ; j < hmm_initial.state_num ; j++)
                    tempSum += alpha[j][t] * beta[j][t];
                for( int i = 0 ; i < hmm_initial.state_num ; i++ ){
                    gammaParam[i][t] = (alpha[i][t] * beta[i][t])  / tempSum;
                }
            }

            // Debug
            // printf("\nAlpha:%.15f Beta:%.15f\t", alpha[0][0], beta[0][0]);
            printf("%.1f ", gammaParam[0][0]);

            // --- ϵ: The probability of transition from state i to state j given observation and model ---
            for(int t = 0 ; t < SEQ_T ; t++ ){
                for( int i = 0 ; i < hmm_initial.state_num ; i++ ){
                    for(int j = 0 ; j < hmm_initial.state_num ; j++){
                        int tempO = (int)sequenceData[currentSequenceLine][t+1] - 65;
                        double denominator = 0.0;

                        // Denominator
                        for(int inner_i = 0 ; inner_i < hmm_initial.state_num ; inner_i++){
                            for(int inner_j = 0 ; inner_j < hmm_initial.state_num ; inner_j++){
                                denominator += (alpha[inner_i][t] * hmm_initial.transition[inner_i][inner_j] * hmm_initial.observation[tempO][inner_j] * beta[inner_j][t+1]);
                            }
                        }

                        // Epsilon
                        epsilon[t][i][j] = (alpha[i][t] * hmm_initial.transition[i][j] * hmm_initial.observation[tempO][j] * beta[j][t+1]) / denominator;
                    }
                }
            }
            
            // --- Update parameters ---
            // 1. π
            for(int i = 0 ; i < hmm_initial.state_num ; i++){
                // new_hmm.initial[i] = gamma[i][0];
                if(currentSequenceLine == 0)
                    new_hmm.initial[i] = gammaParam[i][0];
                else
                    new_hmm.initial[i] += gammaParam[i][0];
            }
            

            // 2. A
            for(int i = 0 ; i < hmm_initial.state_num ; i++ ){
                for(int j = 0 ; j < hmm_initial.state_num ; j++){
                    double epsilon_sum = 0.0;
                    double gamma_sum = 0.0;
                    for(int t = 0; t < SEQ_T-1 ; t++){
                        epsilon_sum += epsilon[t][i][j];
                        gamma_sum += gammaParam[i][t];
                    }
                    // new_hmm.transition[i][j] = epsilon_sum / gamma_sum;
                    if(currentSequenceLine == 0){
                        a_denominator[i][j] = gamma_sum;
                        a_molecular[i][j] = epsilon_sum;
                    }
                    else{
                        a_denominator[i][j] += gamma_sum;
                        a_molecular[i][j] += epsilon_sum;
                    }
                }
            }
            
            // 3. B
            for(int i = 0 ; i < hmm_initial.state_num ; i++ ){
                for(int k = 0 ; k < hmm_initial.observ_num ; k++){
                    double gamma_sum_t = 0.0;
                    double gamma_sum_observ = 0.0;
                    for(int t = 0; t < SEQ_T ; t++){
                        gamma_sum_t += gammaParam[i][t];
                        if (k+65 == (int)sequenceData[currentSequenceLine][t])
                            gamma_sum_observ += gammaParam[i][t];
                    }
                    // new_hmm.observation[k][i] = gamma_sum_observ / gamma_sum_t;
                    if(currentSequenceLine == 0){
                        b_denominator[k][i] =  gamma_sum_t;
                        b_molecular[k][i] = gamma_sum_observ;
                    }
                    else{
                        b_denominator[k][i] +=  gamma_sum_t;
                        b_molecular[k][i] += gamma_sum_observ;
                    }
                }
            }
            
            // Debug
            // for(int i = 0 ; i < hmm_initial.state_num ; i++ ){
            //     printf("%.10f ", new_hmm.initial[i]);
            // }
            // printf("\n\n");
            
            currentSequenceLine++;

            // Debug
            // if(currentSequenceLine == 1)
            //     printf("\nSecond: %f / %f = %f\n", a_molecular[0][0], a_denominator[0][0], a_molecular[0][0]/a_denominator[0][0]);
        }

        

        // Save the result
        for(int i = 0 ; i < hmm_initial.state_num ; i++ ){
            // pi
            new_hmm.initial[i] = new_hmm.initial[i]/currentSequenceLine;
            
            for(int k = 0 ; k < hmm_initial.observ_num ; k++){
                // A
                new_hmm.transition[i][k] = a_molecular[i][k]/a_denominator[i][k];
                // B
                new_hmm.observation[i][k] = b_molecular[i][k]/b_denominator[i][k];
            }
        }

        // Debug
        // printf("Final Gamma:%lf\n", new_hmm.initial[0]);
        printf("\n");
        hmm_initial = new_hmm;
    }

    // Dump trained model
    FILE *dump_fp = open_or_die( argv[4], "w+");
    dumpHMM( dump_fp, &hmm_initial );
    fclose(dump_fp);
    

    return 0;
}

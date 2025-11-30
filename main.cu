// main.cu
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <pthread.h>

// Biblioteca para gerar números aleatórios robustos na GPU
#include <curand_kernel.h>
#include <cuda_runtime.h> // Para funções CUDA como cudaMalloc, cudaMemcpy

// --- Constantes de Configuração ---
#define NUM_PORTFOLIOS 10000000 // 10 Milhões de simulações
#define RISK_FREE_RATE 0.02      // Taxa livre de risco anual (2%)
#define TRADING_DAYS_PER_YEAR 252 // Fator de anualização

// --- Estruturas e Variáveis Globais para Pthreads ---
typedef struct {
    int id;
    int num_assets;
    int num_days;
    double *log_returns;
    double *avg_returns;
    double *covariance_matrix;
    int assets_per_thread;
    int num_threads;
} ThreadData;

ThreadData *h_thread_data;
pthread_barrier_t barrier;

// Variáveis Host para armazenamento de dados
int h_num_assets = 0;
int h_num_days = 0;
char **h_tickers = NULL;
double *h_log_returns = NULL;
double *h_avg_returns = NULL;
double *h_covariance_matrix = NULL;

// Variáveis Host para a melhor solução
double h_best_sharpe = -100.0;
double h_best_return = 0.0;
double h_best_volatility = 0.0;
double *h_best_weights = NULL;

// --- Protótipos CUDA ---
__global__ void setup_kernel(curandState *state, unsigned long long seed);

__global__ void monte_carlo_kernel(
    int num_assets, 
    double *d_avg_returns, 
    double *d_covariance_matrix, 
    double *d_best_sharpe,
    double *d_best_return,
    double *d_best_volatility,
    double *d_best_weights,
    curandState *rng_states
);
// --- Fim Protótipos CUDA ---

// =================================================================
// FUNÇÕES HOST (CPU) PARA CÁLCULO DA COVARIÂNCIA (PTHREADS)
// =================================================================

/**
 * @brief Função de Pthread para calcular a Matriz de Covariância e Retornos Médios.
 */
void *calculate_stats(void *arg) {
    ThreadData *data = (ThreadData *)arg;
    int start_asset = data->id * data->assets_per_thread;
    int end_asset = start_asset + data->assets_per_thread;

    if (data->id == data->num_threads - 1) {
        end_asset = data->num_assets; // Garantir que a última thread pegue o resto
    }

    // 1. Calcular Retornos Médios (Paralelo)
    for (int i = start_asset; i < end_asset; i++) {
        double sum = 0.0;
        for (int t = 0; t < data->num_days; t++) {
            sum += data->log_returns[t * data->num_assets + i];
        }
        data->avg_returns[i] = sum / data->num_days;
    }

    // Espera até que *TODOS* os retornos médios estejam calculados
    pthread_barrier_wait(&barrier);

    // 2. Calcular Covariância (Semi-paralelo, focado na Matriz Inferior)
    for (int i = start_asset; i < end_asset; i++) {
        for (int j = 0; j <= i; j++) { // Itera apenas sobre a matriz inferior (incluindo diagonal)
            double sum_of_products = 0.0;
            for (int t = 0; t < data->num_days; t++) {
                // Índices de retorno para ativo i e ativo j
                double return_i = data->log_returns[t * data->num_assets + i];
                double return_j = data->log_returns[t * data->num_assets + j];
                
                // Cálculo do termo: (R_i - E[R_i]) * (R_j - E[R_j])
                sum_of_products += (return_i - data->avg_returns[i]) * (return_j - data->avg_returns[j]);
            }
            
            // Covariância/Variância = 1/(N-1) * sum_of_products
            double cov = sum_of_products / (data->num_days - 1);
            
            // Armazenar na Matriz de Covariância
            data->covariance_matrix[i * data->num_assets + j] = cov;
            // A matriz é simétrica, preenche o termo espelhado (j, i)
            data->covariance_matrix[j * data->num_assets + i] = cov;
        }
    }

    return NULL;
}


/**
 * @brief Lê o arquivo binário gerado pelo Python.
 */
int read_binary_data(const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "ERRO: Nao foi possivel abrir o arquivo %s. Execute data_fetch.py primeiro.\n", filename);
        return 0;
    }

    // 1. Ler cabeçalho: num_assets e num_days
    if (fread(&h_num_assets, sizeof(int), 1, file) != 1 || 
        fread(&h_num_days, sizeof(int), 1, file) != 1) {
        fprintf(stderr, "ERRO: Falha na leitura do cabecalho do arquivo.\n");
        fclose(file);
        return 0;
    }

    printf("--- Dados de Entrada ---\n");
    printf("Numero de ativos: %d\n", h_num_assets);
    printf("Numero de dias de historico: %d\n", h_num_days);

    // 2. Ler Tickers
    h_tickers = (char**)malloc(h_num_assets * sizeof(char*));
    for (int i = 0; i < h_num_assets; i++) {
        int len;
        if (fread(&len, sizeof(int), 1, file) != 1) return 0;
        h_tickers[i] = (char*)malloc(len + 1);
        if (fread(h_tickers[i], 1, len, file) != len) return 0;
        h_tickers[i][len] = '\0';
        printf("Ativo %d: %s\n", i + 1, h_tickers[i]);
    }

    // 3. Alocar e ler retornos
    h_log_returns = (double*)malloc(h_num_assets * h_num_days * sizeof(double));
    size_t expected_size = h_num_assets * h_num_days * sizeof(double);
    if (fread(h_log_returns, 1, expected_size, file) != expected_size) {
        fprintf(stderr, "ERRO: Falha na leitura dos dados de retorno.\n");
        free(h_log_returns);
        fclose(file);
        return 0;
    }
    
    fclose(file);
    return 1;
}

/**
 * @brief Libera a memória alocada no Host.
 */
void cleanup_host_memory() {
    if (h_log_returns) free(h_log_returns);
    if (h_avg_returns) free(h_avg_returns);
    if (h_covariance_matrix) free(h_covariance_matrix);
    if (h_best_weights) free(h_best_weights);
    if (h_tickers) {
        for (int i = 0; i < h_num_assets; i++) free(h_tickers[i]);
        free(h_tickers);
    }
    if (h_thread_data) free(h_thread_data);
}


// =================================================================
// FUNÇÕES DEVICE (GPU) PARA MONTE CARLO (CUDA KERNEL)
// =================================================================

/**
 * @brief Inicializa os estados do gerador de números aleatórios CURAND.
 */
__global__ void setup_kernel(curandState *state, unsigned long long seed) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    // Inicializa o gerador, usando um seed fixo + o ID do thread
    curand_init(seed, id, 0, &state[id]);
}

/**
 * @brief Kernel CUDA para executar a Simulação Monte Carlo.
 */
__global__ void monte_carlo_kernel(
    int num_assets, 
    double *d_avg_returns, 
    double *d_covariance_matrix, 
    double *d_best_sharpe,
    double *d_best_return,
    double *d_best_volatility,
    double *d_best_weights,
    curandState *rng_states
) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= NUM_PORTFOLIOS) return;

    curandState local_state = rng_states[id];
    
    // Variáveis locais para o cálculo do portfólio
    double weights[10]; 
    if (num_assets > 10) return; // Limitação de exemplo

    double sum_of_weights = 0.0;
    
    // 1. Gerar Pesos Aleatórios e Soma
    for (int i = 0; i < num_assets; i++) {
        weights[i] = curand_uniform_double(&local_state);
        sum_of_weights += weights[i];
    }
    
    // 2. Normalizar Pesos (devem somar 1)
    if (sum_of_weights > 0.0) {
        for (int i = 0; i < num_assets; i++) {
            weights[i] /= sum_of_weights;
        }
    } else {
        return;
    }
    
    // --- Cálculo do Portfólio ---
    double portfolio_return = 0.0;
    double portfolio_variance = 0.0;
    
    // 3. Calcular Retorno do Portfólio (Rp)
    for (int i = 0; i < num_assets; i++) {
        portfolio_return += weights[i] * d_avg_returns[i];
    }
    
    // 4. Calcular Variância do Portfólio (sigma_p^2)
    for (int i = 0; i < num_assets; i++) {
        for (int j = 0; j < num_assets; j++) {
            portfolio_variance += weights[i] * weights[j] * d_covariance_matrix[i * num_assets + j];
        }
    }

    // 5. Anualizar e Calcular Volatilidade
    // CORREÇÃO 1: Usando ::sqrt() ou math.h padrão para a versão device
    portfolio_return *= TRADING_DAYS_PER_YEAR;
    double portfolio_volatility = ::sqrt(portfolio_variance) * ::sqrt(TRADING_DAYS_PER_YEAR); 

    // 6. Calcular Sharpe Ratio
    double sharpe_ratio = (portfolio_return - RISK_FREE_RATE) / portfolio_volatility;

    // --- Otimização Global (Atomic Update) ---

    // 7. Atomicamente tentar atualizar o melhor Sharpe Ratio
    double old_best_sharpe;
    do {
        old_best_sharpe = *d_best_sharpe;
        if (sharpe_ratio <= old_best_sharpe) {
            return;
        }
        // CORREÇÃO 2: Convertendo o ponteiro para (unsigned long long*) para atomicCAS
    } while (atomicCAS((unsigned long long*)d_best_sharpe, 
                       __double_as_longlong(old_best_sharpe), 
                       __double_as_longlong(sharpe_ratio)) != __double_as_longlong(old_best_sharpe));
    
    // Se esta thread venceu a atualização atômica, ela salva os pesos e estatísticas
    
    *d_best_return = portfolio_return;
    *d_best_volatility = portfolio_volatility;
    
    for (int i = 0; i < num_assets; i++) {
        d_best_weights[i] = weights[i];
    }
    
    rng_states[id] = local_state;
}

// =================================================================
// FUNÇÃO PRINCIPAL (HOST)
// =================================================================

int main() {
    // 0. Pré-requisitos
    if (!read_binary_data("log_returns.bin")) {
        cleanup_host_memory();
        return 1;
    }

    // 1. Alocação de Memória do Host
    h_avg_returns = (double*)malloc(h_num_assets * sizeof(double));
    h_covariance_matrix = (double*)malloc(h_num_assets * h_num_assets * sizeof(double));
    h_best_weights = (double*)malloc(h_num_assets * sizeof(double));
    
    if (!h_avg_returns || !h_covariance_matrix || !h_best_weights) {
        fprintf(stderr, "ERRO: Falha na alocacao de memoria do Host.\n");
        cleanup_host_memory();
        return 1;
    }

    // 2. Cálculo da Matriz de Covariância na CPU (Pthreads)
    const int NUM_CPU_THREADS = 4;
    pthread_t threads[NUM_CPU_THREADS];
    h_thread_data = (ThreadData*)malloc(NUM_CPU_THREADS * sizeof(ThreadData));
    int assets_per_thread = h_num_assets / NUM_CPU_THREADS;
    
    if (pthread_barrier_init(&barrier, NULL, NUM_CPU_THREADS) != 0) {
        fprintf(stderr, "ERRO: Falha ao inicializar a barreira Pthread.\n");
        cleanup_host_memory();
        return 1;
    }
    
    printf("\n--- Calculo da Matriz de Covariancia (Pthreads) ---\n");
    clock_t start_cpu = clock();

    for (int i = 0; i < NUM_CPU_THREADS; i++) {
        h_thread_data[i].id = i;
        h_thread_data[i].num_assets = h_num_assets;
        h_thread_data[i].num_days = h_num_days;
        h_thread_data[i].log_returns = h_log_returns;
        h_thread_data[i].avg_returns = h_avg_returns;
        h_thread_data[i].covariance_matrix = h_covariance_matrix;
        h_thread_data[i].assets_per_thread = assets_per_thread;
        h_thread_data[i].num_threads = NUM_CPU_THREADS;

        if (pthread_create(&threads[i], NULL, calculate_stats, &h_thread_data[i]) != 0) {
            fprintf(stderr, "ERRO: Falha ao criar Pthread %d.\n", i);
            break; 
        }
    }

    for (int i = 0; i < NUM_CPU_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    
    clock_t end_cpu = clock();
    double cpu_time_used = ((double) (end_cpu - start_cpu)) / CLOCKS_PER_SEC;
    printf("Cálculo Pthreads concluído em: %.4f segundos.\n", cpu_time_used);

    // 3. Inicialização e Alocação CUDA (Device)
    
    double *d_avg_returns = NULL;
    double *d_covariance_matrix = NULL;
    double *d_best_sharpe = NULL;
    double *d_best_return = NULL;
    double *d_best_volatility = NULL;
    double *d_best_weights = NULL;
    curandState *d_rng_states = NULL;

    size_t avg_size = h_num_assets * sizeof(double);
    size_t cov_size = h_num_assets * h_num_assets * sizeof(double);
    size_t weights_size = h_num_assets * sizeof(double);
    size_t rng_size = NUM_PORTFOLIOS * sizeof(curandState);

    // Alocação
    cudaMalloc((void**)&d_avg_returns, avg_size);
    cudaMalloc((void**)&d_covariance_matrix, cov_size);
    cudaMalloc((void**)&d_best_sharpe, sizeof(double));
    cudaMalloc((void**)&d_best_return, sizeof(double));
    cudaMalloc((void**)&d_best_volatility, sizeof(double));
    cudaMalloc((void**)&d_best_weights, weights_size);
    cudaMalloc((void**)&d_rng_states, rng_size);

    // 4. Transferência de Dados Host -> Device (CUDA Memcpy)
    printf("\n--- Transferencia Host (CPU) -> Device (GPU) ---\n");
    cudaMemcpy(d_avg_returns, h_avg_returns, avg_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_covariance_matrix, h_covariance_matrix, cov_size, cudaMemcpyHostToDevice);

    double initial_sharpe = -1000.0;
    cudaMemcpy(d_best_sharpe, &initial_sharpe, sizeof(double), cudaMemcpyHostToDevice);

    // 5. Lançamento dos Kernels CUDA
    printf("\n--- Execucao Monte Carlo (CUDA Kernel) ---\n");
    printf("Lancando %d simulacoes...\n", NUM_PORTFOLIOS);

    int threads_per_block = 256;
    int blocks = (NUM_PORTFOLIOS + threads_per_block - 1) / threads_per_block;
    
    // Inicializar estados PRNG
    setup_kernel<<<blocks, threads_per_block>>>(d_rng_states, (unsigned long long)time(NULL));
    cudaDeviceSynchronize();
    
    clock_t start_gpu = clock();

    // Lançamento do Kernel Monte Carlo
    monte_carlo_kernel<<<blocks, threads_per_block>>>(
        h_num_assets, 
        d_avg_returns, 
        d_covariance_matrix, 
        d_best_sharpe, 
        d_best_return, 
        d_best_volatility, 
        d_best_weights, 
        d_rng_states
    );
    
    cudaDeviceSynchronize(); // Espera a GPU terminar

    clock_t end_gpu = clock();
    double gpu_time_used = ((double) (end_gpu - start_gpu)) / CLOCKS_PER_SEC;
    printf("Simulacao CUDA concluida em: %.4f segundos.\n", gpu_time_used);

    // 6. Transferência de Resultados Device -> Host
    cudaMemcpy(&h_best_sharpe, d_best_sharpe, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_best_return, d_best_return, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_best_volatility, d_best_volatility, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_best_weights, d_best_weights, weights_size, cudaMemcpyDeviceToHost);

    // 7. Exibir Resultados
    printf("\n--- Resultado da Otimizacao ---\n");
    printf("Portfólio com melhor Sharpe Ratio (SR):\n");
    printf(" Sharpe Ratio: %.4f\n", h_best_sharpe);
    printf(" Retorno Anualizado: %.2f%%\n", h_best_return * 100);
    printf(" Volatilidade Anualizada: %.2f%%\n", h_best_volatility * 100);
    printf(" Pesos:\n");
    for (int i = 0; i < h_num_assets; i++) {
        printf("  - %s: %.2f%%\n", h_tickers[i], h_best_weights[i] * 100);
    }
    printf("\nPerformance: CPU (Covariance): %.4f s | GPU (Monte Carlo): %.4f s\n", cpu_time_used, gpu_time_used);

    // 8. Limpeza
    cudaFree(d_avg_returns);
    cudaFree(d_covariance_matrix);
    cudaFree(d_best_sharpe);
    cudaFree(d_best_return);
    cudaFree(d_best_volatility);
    cudaFree(d_best_weights);
    cudaFree(d_rng_states);
    pthread_barrier_destroy(&barrier);
    cleanup_host_memory();

    return 0;
}
// main_seq.c (Versão Sequencial para Comparação)
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

// --- Constantes de Configuração ---
// Ajustado para 10.000.000 (10 Milhões) de simulações para corresponder ao main.cu
#define NUM_PORTFOLIOS 10000000 
#define RISK_FREE_RATE 0.02 // Taxa livre de risco anual (2%)
#define TRADING_DAYS_PER_YEAR 252 // Fator de anualização

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

// =================================================================
// FUNÇÕES SEQUENCIAIS (CPU)
// =================================================================

/**
 * @brief Calcula Retornos Médios e Matriz de Covariância em modo sequencial.
 */
void calculate_stats_sequential() {
	printf("1. Calculando Retornos Medios e Matriz de Covariancia (CPU Sequencial)...\n");

	// 1. Calcular Retornos Médios
	for (int i = 0; i < h_num_assets; i++) {
		double sum = 0.0;
		for (int t = 0; t < h_num_days; t++) {
			// h_log_returns é a matriz plana (num_days x num_assets)
			sum += h_log_returns[t * h_num_assets + i];
		}
		h_avg_returns[i] = sum / h_num_days;
	}

	// 2. Calcular Covariância (Sequencial)
	for (int i = 0; i < h_num_assets; i++) {
		for (int j = 0; j < h_num_assets; j++) {
			double sum_of_products = 0.0;
			for (int t = 0; t < h_num_days; t++) {
				double return_i = h_log_returns[t * h_num_assets + i];
				double return_j = h_log_returns[t * h_num_assets + j];
				
				sum_of_products += (return_i - h_avg_returns[i]) * (return_j - h_avg_returns[j]);
			}
			
			// Covariância/Variância = 1/(N-1) * sum_of_products
			double cov = sum_of_products / (h_num_days - 1);
			
			h_covariance_matrix[i * h_num_assets + j] = cov;
		}
	}
}

/**
 * @brief Simulação Monte Carlo sequencial para otimização do portfólio.
 */
void run_monte_carlo_sequential() {
	printf("2. Executando Simulação Monte Carlo (%d portfólios) (CPU Sequencial)...\n", NUM_PORTFOLIOS);

	srand(time(NULL)); // Inicializa o gerador de números aleatórios
	h_best_sharpe = -100.0; // Reinicia o melhor Sharpe

	for (int k = 0; k < NUM_PORTFOLIOS; k++) {
		double weights[10]; // Limite de 10 ativos
		if (h_num_assets > 10) return;

		double sum_of_weights = 0.0;
		
		// 1. Gerar Pesos Aleatórios e Soma
		for (int i = 0; i < h_num_assets; i++) {
			weights[i] = (double)rand() / RAND_MAX; // Gera número entre 0 e 1
			sum_of_weights += weights[i];
		}
		
		// 2. Normalizar Pesos
		if (sum_of_weights > 0.0) {
			for (int i = 0; i < h_num_assets; i++) {
				weights[i] /= sum_of_weights;
			}
		} else {
			continue;
		}
		
		// --- Cálculo do Portfólio ---
		double portfolio_return = 0.0;
		double portfolio_variance = 0.0;
		
		// 3. Calcular Retorno (Rp)
		for (int i = 0; i < h_num_assets; i++) {
			portfolio_return += weights[i] * h_avg_returns[i];
		}
		
		// 4. Calcular Variância (sigma_p^2)
		for (int i = 0; i < h_num_assets; i++) {
			for (int j = 0; j < h_num_assets; j++) {
				portfolio_variance += weights[i] * weights[j] * h_covariance_matrix[i * h_num_assets + j];
			}
		}

		// 5. Anualizar e Calcular Volatilidade
		portfolio_return *= TRADING_DAYS_PER_YEAR;
		double portfolio_volatility = sqrt(portfolio_variance) * sqrt(TRADING_DAYS_PER_YEAR); 

		// 6. Calcular Sharpe Ratio
		double sharpe_ratio = (portfolio_return - RISK_FREE_RATE) / portfolio_volatility;

		// 7. Otimização (Comparação Simples Sequencial)
		if (sharpe_ratio > h_best_sharpe) {
			h_best_sharpe = sharpe_ratio;
			h_best_return = portfolio_return;
			h_best_volatility = portfolio_volatility;
			
			for (int i = 0; i < h_num_assets; i++) {
				h_best_weights[i] = weights[i];
			}
		}
	}
}

// =================================================================
// FUNÇÕES AUXILIARES (IGUAIS AO CÓDIGO PARALELO)
// =================================================================

// Função de leitura do arquivo binário (copiada do main.cu)
int read_binary_data(const char *filename) {
	FILE *file = fopen(filename, "rb");
	if (!file) {
		fprintf(stderr, "ERRO: Nao foi possivel abrir o arquivo %s. Execute data_fetch.py primeiro.\n", filename);
		return 0;
	}

	if (fread(&h_num_assets, sizeof(int), 1, file) != 1 || 
		fread(&h_num_days, sizeof(int), 1, file) != 1) {
		fprintf(stderr, "ERRO: Falha na leitura do cabecalho do arquivo.\n");
		fclose(file);
		return 0;
	}

	printf("--- Dados de Entrada ---\n");
	printf("Numero de ativos: %d\n", h_num_assets);
	printf("Numero de dias de historico: %d\n", h_num_days);

	h_tickers = (char**)malloc(h_num_assets * sizeof(char*));
	for (int i = 0; i < h_num_assets; i++) {
		int len;
		if (fread(&len, sizeof(int), 1, file) != 1) return 0;
		h_tickers[i] = (char*)malloc(len + 1);
		if (fread(h_tickers[i], 1, len, file) != len) return 0;
		h_tickers[i][len] = '\0';
		printf("Ativo %d: %s\n", i + 1, h_tickers[i]);
	}

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

// Função de limpeza de memória (copiada do main.cu)
void cleanup_host_memory() {
	if (h_log_returns) free(h_log_returns);
	if (h_avg_returns) free(h_avg_returns);
	if (h_covariance_matrix) free(h_covariance_matrix);
	if (h_best_weights) free(h_best_weights);
	if (h_tickers) {
		for (int i = 0; i < h_num_assets; i++) free(h_tickers[i]);
		free(h_tickers);
	}
}

// =================================================================
// MAIN
// =================================================================

int main() {
	// 0. Leitura de Dados
	if (!read_binary_data("log_returns.bin")) {
		cleanup_host_memory();
		return 1;
	}

	// 1. Alocação de Memória
	h_avg_returns = (double*)malloc(h_num_assets * sizeof(double));
	h_covariance_matrix = (double*)malloc(h_num_assets * h_num_assets * sizeof(double));
	h_best_weights = (double*)malloc(h_num_assets * sizeof(double));
	
	if (!h_avg_returns || !h_covariance_matrix || !h_best_weights) {
		fprintf(stderr, "ERRO: Falha na alocacao de memoria do Host.\n");
		cleanup_host_memory();
		return 1;
	}

	// --- EXECUÇÃO SEQUENCIAL ---
	clock_t start_total = clock();

	// 2. Cálculo de Estatísticas (Covariância)
	calculate_stats_sequential();

	// 3. Simulação Monte Carlo
	run_monte_carlo_sequential();
	
	clock_t end_total = clock();
	double cpu_time_used = ((double) (end_total - start_total)) / CLOCKS_PER_SEC;
	// --------------------------

	// 4. Exibir Resultados
	printf("\n--- Resultado da Otimizacao SEQUENCIAL ---\n");
	printf("Portfólio com melhor Sharpe Ratio (SR):\n");
	printf(" Sharpe Ratio: %.4f\n", h_best_sharpe);
	printf(" Retorno Anualizado: %.2f%%\n", h_best_return * 100);
	printf(" Volatilidade Anualizada: %.2f%%\n", h_best_volatility * 100);
	printf(" Pesos:\n");
	for (int i = 0; i < h_num_assets; i++) {
		printf("  - %s: %.2f%%\n", h_tickers[i], h_best_weights[i] * 100);
	}
	printf("\nTempo Total de Execução (CPU Sequencial - %d simulações): %.4f segundos.\n", NUM_PORTFOLIOS, cpu_time_used);

	// 5. Limpeza
	cleanup_host_memory();

	return 0;
}
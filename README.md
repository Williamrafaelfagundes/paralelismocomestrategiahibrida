Otimização de Portfólios Híbrida (CPU/GPU)
Este projeto demonstra a otimização de portfólios financeiros usando uma arquitetura de computação de alto desempenho (HPC) que combina Pthreads (CPU) e CUDA (GPU). O objetivo é encontrar a melhor alocação de pesos de ativos que maximiza o Sharpe Ratio (retorno ajustado ao risco).

💡 Como FuncionaO projeto divide o trabalho de processamento em três etapas:Python (data_fetch.py): Baixa dados de ações e calcula os retornos logarítmicos.CPU Paralela (Pthreads): Calcula a Matriz de Covariância $\mathbf{\Sigma}$ e os Retornos Médios (tarefas de pré-processamento).GPU Paralela (CUDA): Executa uma simulação Monte Carlo massiva (10 milhões de portfólios) para encontrar o portfólio ideal de forma extremamente rápida.

⚙️ Configuração e Execução
Siga os passos abaixo para instalar as dependências, compilar o código e rodar o otimizador.

1. Instalar Dependências
Certifique-se de ter o Python 3 e o CUDA Toolkit da NVIDIA instalados.

Bash
# Instala as bibliotecas Python (yfinance, numpy)
pip3 install yfinance numpy

Claro! Aqui está uma descrição simples e direta para o seu README.md, focada no propósito do projeto e nas instruções para rodá-lo, como você pediu.

🚀 Otimização de Portfólios Híbrida (CPU/GPU)
Este projeto demonstra a otimização de portfólios financeiros usando uma arquitetura de computação de alto desempenho (HPC) que combina Pthreads (CPU) e CUDA (GPU). O objetivo é encontrar a melhor alocação de pesos de ativos que maximiza o Sharpe Ratio (retorno ajustado ao risco).

💡 Como Funciona
O projeto divide o trabalho de processamento em três etapas:

Python (data_fetch.py): Baixa dados de ações e calcula os retornos logarítmicos.

CPU Paralela (Pthreads): Calcula a Matriz de Covariância Σ e os Retornos Médios (tarefas de pré-processamento).

GPU Paralela (CUDA): Executa uma simulação Monte Carlo massiva (10 milhões de portfólios) para encontrar o portfólio ideal de forma extremamente rápida.


Shutterstock
Explorar
⚙️ Configuração e Execução
Siga os passos abaixo para instalar as dependências, compilar o código e rodar o otimizador.

1. Instalar Dependências
Certifique-se de ter o Python 3 e o CUDA Toolkit da NVIDIA instalados.

Bash
# Instala as bibliotecas Python (yfinance, numpy)
pip3 install yfinance numpy

2. Gerar o Arquivo de Dados
O script Python baixará os dados dos ativos e criará o arquivo binário log_returns.bin.

Bash
python3 data_fetch.py

3. Compilar o Projeto
Use o compilador nvcc para compilar o código C/CUDA, incluindo as flags para Pthreads e a biblioteca de números aleatórios (curand).

Bash
nvcc --expt-relaxed-constexpr main.cu -o portfolio -Xcompiler -pthread -lcurand

4. Executar a Otimização
O executável lerá os dados, fará o pré-processamento na CPU e executará a otimização massiva na GPU, exibindo o resultado final:

Bash
./portfolio

🎯 Resultado Esperado
O programa exibirá o tempo de processamento para CPU e GPU, além das métricas do portfólio vencedor:

--- Resultado da Otimizacao ---
Portfólio com melhor Sharpe Ratio (SR):
 Sharpe Ratio: X.XXXX
 Retorno Anualizado: XX.XX%
 Volatilidade Anualizada: XX.XX%
 Pesos:
  - AAPL: XX.XX%
  - GOOGL: XX.XX%
  - MSFT: XX.XX%
  ...
Performance: CPU (Covariance): X.XXXX s | GPU (Monte Carlo): X.XXXX s

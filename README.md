🎓 Otimização de Portfólios de Ações: Comparativo Paralelo vs. Sequencial
Este projeto foi desenvolvido como parte de um trabalho de faculdade focado em Computação de Alto Desempenho (HPC). O objetivo principal é demonstrar o ganho de performance ao executar tarefas financeiras complexas usando o paralelismo (CPU e GPU) em comparação com uma execução tradicional sequencial (CPU pura).

A tarefa em questão é a Otimização de Portfólios de Ações através da Simulação Monte Carlo.

💡 A Grande Ideia (Descomplicada)
- Problema: Encontrar a melhor combinação de pesos para 5 ações que oferece o maior retorno ajustado ao risco (Sharpe Ratio).
- Solução: Simular milhões de portfólios aleatórios e comparar os resultados.
- Comparativo:Versão Sequencial:
   A CPU faz todos os passos, um de cada vez.
   Versão Paralela/Híbrida: A CPU usa múltiplos núcleos (Pthreads) para o pré-processamento (Matriz de Covariância), e a GPU (CUDA) usa milhares de núcleos simultâneos para a simulação Monte Carlo.

  O resultado esperado é um ganho massivo de velocidade na versão Paralela, que consegue analisar 10 milhões de portfólios em frações de segundo.

⚙️ Guia Rápido: Como Rodar o ProjetoSiga estes 3 passos simples para rodar ambas as versões e fazer a sua comparação.

1. Preparação: Baixar os Dados (Python)
Este passo baixa o histórico de preços dos 5 ativos e cria o arquivo log_returns.bin, necessário para as versões C/C++ rodarem.

Bash#
Baixa dados, calcula retornos e gera o arquivo binário
python3 data_fetch.py

3. Rodar a Versão Sequencial (CPU Pura)
Esta versão compila e executa o código que faz o trabalho um passo de cada vez.
Compilar g++ main_seq.c -o portfolio_seq -lm

Executar ./portfolio_seq

Resultado: Você verá o tempo total de execução na CPU (em segundos) para 1 milhão de simulações.

3. Rodar a Versão Paralela (CPU + GPU)
  
Esta versão compila e executa o código que divide o trabalho entre CPU (Covariância) e GPU (Monte Carlo).

Compilar nvcc --expt-relaxed-constexpr main.cu -o portfolio -Xcompiler -pthread -lcurand
Executar ./portfolio

Resultado: Você verá os tempos de execução separados para a CPU e a GPU, que será muito mais rápido (em milissegundos) para 10 milhões de simulações.

📈 Conclusão da AnáliseAo comparar os tempos de execução, o projeto demonstra de forma clara a importância e a eficiência da computação paralela para resolver problemas complexos na área de finanças.

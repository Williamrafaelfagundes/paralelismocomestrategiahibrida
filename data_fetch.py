# data_fetch.py
import yfinance as yf
import numpy as np
import datetime as dt
import struct

# Tickers para otimização - AGORA COM 5 ATIVOS
TICKERS = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"] # Adicionados AMZN e TSLA
START_DATE = (dt.datetime.now() - dt.timedelta(days=365 * 5)).strftime('%Y-%m-%d') # Últimos 5 anos
END_DATE = dt.datetime.now().strftime('%Y-%m-%d')
OUTPUT_FILE = "log_returns.bin"

def fetch_data():
    """Baixa os dados e calcula os retornos logarítmicos diários."""
    print(f"Baixando dados para: {TICKERS} de {START_DATE} a {END_DATE}...")
    
    # 1. Baixar dados ajustados de fechamento
    # Usamos auto_adjust=False para garantir que a coluna 'Adj Close' seja retornada.
    data = yf.download(TICKERS, start=START_DATE, end=END_DATE, auto_adjust=False)['Adj Close']
    
    # 2. Calcular Retornos Diários Logarítmicos
    log_returns = np.log(data / data.shift(1)).dropna()
    
    print(f"Dados obtidos. {len(log_returns)} dias de retornos.")
    
    # 3. Preparar dados para gravação binária
    num_assets = len(TICKERS)
    num_days = len(log_returns)
    
    # Converter a matriz de retornos para um array 1D no formato C (row-major)
    returns_flat = log_returns.values.astype(np.float64).flatten()
    
    # 4. Gravar no arquivo binário
    with open(OUTPUT_FILE, 'wb') as f:
        # Gravar número de ativos e número de dias
        f.write(struct.pack('ii', num_assets, num_days))
        
        # Gravar os tickers (5 nomes)
        for ticker in TICKERS:
            f.write(struct.pack('i', len(ticker)))
            f.write(ticker.encode('utf-8'))
            
        # Gravar os retornos
        f.write(returns_flat.tobytes())
        
    print(f"Dados de retornos salvos com sucesso em {OUTPUT_FILE}.")

if __name__ == "__main__":
    fetch_data()
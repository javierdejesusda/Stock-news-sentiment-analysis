import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional

sns.set_theme(style="whitegrid")

def plot_sentiment_vs_price(
    sentiment_df: pd.DataFrame, 
    price_df: pd.DataFrame, 
    ticker: str,
    sentiment_col: str, # Columna a usar (ej. 'vader_mean' o 'finbert_mean')
    sentiment_label: str, # Etiqueta para el gráfico (ej. 'Sentimiento VADER')
    output_file: Optional[str] = None
):
    
    sentiment_df.index = pd.to_datetime(sentiment_df.index)
    price_df.index = pd.to_datetime(price_df.index)

    # Unir usando la columna de sentimiento especificada
    combined_df = price_df.join(sentiment_df[sentiment_col], how='inner')
    
    if combined_df.empty:
        print(f"Visualización OMITIDA: No hay datos superpuestos para {ticker} y {sentiment_label}.")
        return

    fig, ax1 = plt.subplots(figsize=(15, 7))
    
    color = 'tab:blue'
    ax1.set_xlabel('Fecha')
    ax1.set_ylabel(f'Precio Cierre {ticker}', color=color)
    ax1.plot(combined_df.index, combined_df['close_price'], color=color, label=f'Precio {ticker}')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel(sentiment_label, color=color)  
    # Plotear usando la columna de sentimiento especificada
    ax2.plot(combined_df.index, combined_df[sentiment_col], color=color, linestyle='--', label=sentiment_label)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.axhline(0, color='grey', lw=0.5, linestyle=':') 

    fig.tight_layout()  
    plt.title(f'{sentiment_label} vs. Precio de Cierre para {ticker}')
    
    if output_file:
        plt.savefig(output_file)
    plt.show()

def plot_sentiment_trends(
    sentiment_df: pd.DataFrame, 
    output_file: Optional[str] = None
):
    if sentiment_df.empty:
        return

    plt.figure(figsize=(15, 7))
    
    sentiment_df.index = pd.to_datetime(sentiment_df.index)
    
    sns.lineplot(data=sentiment_df, x=sentiment_df.index, y='vader_mean', label='VADER (Media)')
    sns.lineplot(data=sentiment_df, x=sentiment_df.index, y='finbert_mean', label='FinBERT (Media)')
    
    plt.axhline(0, color='grey', lw=0.5, linestyle=':')
    plt.title('Tendencia del Sentimiento Medio Diario (VADER vs. FinBERT)')
    plt.xlabel('Fecha')
    plt.ylabel('Puntuación Media de Sentimiento')
    plt.legend()
    
    if output_file:
        plt.savefig(output_file)
    plt.show()

def plot_model_comparison(
    scores_df: pd.DataFrame, 
    output_file: Optional[str] = None
):
    if scores_df.empty:
        return

    plt.figure(figsize=(10, 7))
    
    sns.scatterplot(
        data=scores_df, 
        x='vader_score', 
        y='finbert_score', 
        alpha=0.3,
        s=15 
    )
    
    plt.axhline(0, color='grey', lw=0.5, linestyle=':')
    plt.axvline(0, color='grey', lw=0.5, linestyle=':')
    plt.title('Comparación de Puntuaciones por Noticia (VADER vs. FinBERT)')
    plt.xlabel('Puntuación VADER (Compound)')
    plt.ylabel('Puntuación FinBERT (Mapeada)')
    
    if output_file:
        plt.savefig(output_file)
    plt.show()
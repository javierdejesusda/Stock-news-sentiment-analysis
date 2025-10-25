import sys
# import datetime
from typing import Optional

import pandas as pd
# import nltk 

import config
from api_fetcher import AlphaVantageFetcher
from sentiment_processor import FinancialSentimentProcessor


# def setup_nltk() -> None:
#     try:
#         nltk.data.find('sentiment/vader_lexicon.zip')
#     except LookupError:
#         nltk.download('vader_lexicon')

# def analizar_comparativa_sentimiento(scores_df: pd.DataFrame) -> None:
#     """
#     Realiza un análisis comparativo entre VADER y FinBERT.
#     """
#     if scores_df.empty or 'vader_score' not in scores_df or 'finbert_score' not in scores_df:
#         print("DataFrame de puntuaciones vacío o incompleto.")
#         return
    
#     print("\nEstadísticas Descriptivas")
#     desc_stats = scores_df[['vader_score', 'finbert_score']].describe()
#     print(f"\n{desc_stats.to_string()}")

#     print("\n Matriz de Correlación ")
#     correlation = scores_df[['vader_score', 'finbert_score']].corr()
#     print(f"\n{correlation.to_string()}")

#     print("\n Desacuerdos ")
#     scores_df['score_diff'] = (scores_df['vader_score'] - scores_df['finbert_score']).abs()
#     top_5_disagreements = scores_df.nlargest(5, 'score_diff')
    
#     for _, row in top_5_disagreements.iterrows():
#         print(f"\nDesacuerdo (Diff: {row['score_diff']:.4f})")
#         print(f"  -> VADER:   {row['vader_score']:.4f}")
#         print(f"  -> FinBERT: {row['finbert_score']:.4f}")
#         print(f"  -> Texto:     {row['texto_noticia'][:150]}...")
#         print(f"  -> URL:       {row['url']}")

def run_pipeline() -> None:
    """
    Función principal que ejecuta el pipeline completo.
    """
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    API_KEY: Optional[str] = config.ALPHA_VANTAGE_API_KEY
    if not API_KEY or API_KEY == "TU_CLAVE_API_VA_AQUI":
        print("ALPHA_VANTAGE_API_KEY no está configurada en .env.")
        sys.exit(1)

    # Definir parámetros de consulta
    tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'NVDA']
    # topics = ['earnings', 'financial_markets', 'economy_monetary']
    # time_from = (datetime.datetime.now() - datetime.timedelta(days=7)).strftime('%Y%m%dT%H%M')

    try:
        fetcher = AlphaVantageFetcher(api_key=API_KEY)
        
        news_df = fetcher.fetch_news(
            tickers=tickers
            # topics=topics
            # time_from=time_from
        )

        if news_df.empty:
            print("\nNo se obtuvieron noticias.")
            return

        # Procesar Sentimiento 
        processor = FinancialSentimentProcessor(
            finbert_model_name=config.FINBERT_MODEL_NAME
        )
        scores_df = processor.calculate_sentiments(news_df)

        final_features_df = processor.aggregate_features(scores_df)

        if not final_features_df.empty:
    
            print("\n DataFrame 'sentimiento' (Features Agregadas Diarias)")
            print(f"\n{final_features_df.to_string()}")
            
            # Guardar en CSV 
            # final_features_df.to_csv("sentimiento_agregado_diario.csv")

        else:
            print("No se generaron features agregadas (DataFrame vacío).")


    except Exception as e:
        print(f"\n ERROR INESPERADO: {e}")
        sys.exit(1)
        
        


if __name__ == "__main__":
    # setup_nltk()
    run_pipeline()
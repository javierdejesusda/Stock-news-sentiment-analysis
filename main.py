import sys
import os 
import pandas as pd
import config
from api_fetcher import AlphaVantageFetcher
from sentiment_processor import FinancialSentimentProcessor
import visualizer 

NEWS_CACHE_PATH = "news_data_cache.csv"
SCORES_CACHE_PATH = "scores_data_cache.csv"
PRICE_CACHE_TEMPLATE = "price_data_{}.csv" 

def run_pipeline() -> None:
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    API_KEY = config.ALPHA_VANTAGE_API_KEY
    if not API_KEY:
        print("ALPHA_VANTAGE_API_KEY no está configurada en .env.")
        sys.exit(1)

    tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'NVDA'] 
    main_ticker_for_plot = tickers[4] 

    try:
        fetcher = AlphaVantageFetcher(api_key=API_KEY)

        if os.path.exists(NEWS_CACHE_PATH):
            news_df = pd.read_csv(NEWS_CACHE_PATH, parse_dates=['fecha'])
        else:
            news_df = fetcher.fetch_news(tickers=tickers)
            if not news_df.empty:
                news_df.to_csv(NEWS_CACHE_PATH, index=False)
        
        if news_df.empty:
            print("\nNo se obtuvieron noticias.")
            return

        processor = FinancialSentimentProcessor(
            finbert_model_name=config.FINBERT_MODEL_NAME
        )
        
        if os.path.exists(SCORES_CACHE_PATH):
            scores_df = pd.read_csv(SCORES_CACHE_PATH, parse_dates=['fecha'])
        else:
            scores_df = processor.calculate_sentiments(news_df)
            if not scores_df.empty:
                scores_df.to_csv(SCORES_CACHE_PATH, index=False)

        if scores_df.empty:
            print("\nNo se pudieron calcular las puntuaciones de sentimiento.")
            return

        PRICE_CACHE_PATH = PRICE_CACHE_TEMPLATE.format(main_ticker_for_plot)
        
        if os.path.exists(PRICE_CACHE_PATH):
            price_df = pd.read_csv(PRICE_CACHE_PATH, parse_dates=['fecha'], index_col='fecha')
        else:
            price_df = fetcher.fetch_stock_prices(main_ticker_for_plot) 
            if not price_df.empty:
                price_df.to_csv(PRICE_CACHE_PATH)

        if price_df.empty:
            print(f"\nNo se obtuvieron datos de precio para {main_ticker_for_plot}.")
            
        final_features_df = processor.aggregate_features(scores_df)

        if not final_features_df.empty:
            print("\n DataFrame 'sentimiento' (Features Agregadas Diarias)")
            print(f"\n{final_features_df.head().to_string()}")
        else:
            print("No se generaron features agregadas (DataFrame vacío).")

        visualizer.plot_sentiment_vs_price(
            final_features_df, 
            price_df, 
            main_ticker_for_plot,
            sentiment_col='finbert_mean', 
            sentiment_label='Sentimiento Medio FinBERT',
            output_file="plot_precio_vs_sentimiento_FINBERT.png"
        )
        
        visualizer.plot_sentiment_vs_price(
            final_features_df, 
            price_df, 
            main_ticker_for_plot,
            sentiment_col='vader_mean',  
            sentiment_label='Sentimiento Medio VADER',
            output_file="plot_precio_vs_sentimiento_VADER.png"
        )
        
        visualizer.plot_sentiment_trends(
            final_features_df, 
            output_file="plot_tendencias_sentimiento.png"
        )

    except Exception as e:
        print(f"\n ERROR INESPERADO: {e}")
        sys.exit(1)
        
if __name__ == "__main__":
    run_pipeline()
from typing import List, Optional, Dict, Any

import pandas as pd
import requests
from requests.exceptions import RequestException

class AlphaVantageFetcher:
    BASE_URL: str = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("La clave de la API no puede estar vacía.")
        self.api_key = api_key

    def fetch_news(
        self,
        topics: Optional[List[str]] = None,
        tickers: Optional[List[str]] = None,
        time_from: Optional[str] = None,
        time_to: Optional[str] = None
    ) -> pd.DataFrame:
        
        params: Dict[str, Any] = {
            "function": "NEWS_SENTIMENT",
            "apikey": self.api_key,
            "limit": 1000  
        }

        if tickers:
            params["tickers"] = ",".join(tickers)
        if topics:
            params["topics"] = ",".join(topics)
        if time_from:
            params["time_from"] = time_from
        if time_to:
            params["time_to"] = time_to
        
        empty_df = pd.DataFrame(columns=['fecha', 'texto_noticia', 'source', 'url'])

        try:
            response = requests.get(self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()

            if "feed" not in data or not data["feed"]:
                return empty_df

            feed = data['feed']
            news_items = []

            for item in feed:
                title = item.get('title', '')
                summary = item.get('summary', '')
                texto_noticia = f"{title}. {summary}"
                time_published = item.get('time_published')

                if not time_published or not texto_noticia.strip(". "):
                    continue

                news_items.append({
                    'time_published': time_published,
                    'texto_noticia': texto_noticia,
                    'source': item.get('source'),
                    'url': item.get('url')
                })

            if not news_items:
                return empty_df

            df = pd.DataFrame(news_items)
            
            df['fecha'] = pd.to_datetime(df['time_published'], format='%Y%m%dT%H%M%S')
            df['fecha'] = df['fecha'].dt.normalize()

            return df[['fecha', 'texto_noticia', 'source', 'url']]

        except (RequestException, KeyError, TypeError, ValueError):
            return empty_df

    def fetch_stock_prices(self, ticker: str) -> pd.DataFrame:
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": ticker,
            "apikey": self.api_key,
            "outputsize": "full"
        }
        
        empty_df = pd.DataFrame(columns=['fecha', 'close_price']).set_index('fecha')

        try:
            response = requests.get(self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()

            if "Time Series (Daily)" not in data:
                return empty_df

            time_series = data["Time Series (Daily)"]
            price_data = []
            for date_str, values in time_series.items():
                price_data.append({
                    "fecha": date_str,
                    "close_price": float(values["4. close"])
                })
            
            if not price_data:
                return empty_df

            df = pd.DataFrame(price_data)
            df['fecha'] = pd.to_datetime(df['fecha'])
            df = df.set_index('fecha').sort_index()
            
            return df

        except (RequestException, KeyError, TypeError, ValueError):
            return empty_df
from typing import List, Optional, Dict, Any

import pandas as pd
import requests
from requests.exceptions import RequestException, JSONDecodeError

class AlphaVantageFetcher:
    """
    Clase dedicada a interactuar con la API de Alpha Vantage
    para obtener noticias y sentimiento del mercado.
    """
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
        
        empty_df = pd.DataFrame(columns=['fecha', 'texto_noticia'])

        try:
            response = requests.get(self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()

            if "feed" not in data or not data["feed"]:
                if "Information" in data:
                    print(f"API Info: {data.get('Information')}")
                else:
                    print("La API no devolvió artículos.")
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
                print("No se encontraron artículos válidos.")
                return empty_df

            df = pd.DataFrame(news_items)
            
            # Convertir y normalizar la fecha
            df['fecha'] = pd.to_datetime(df['time_published'], format='%Y%m%dT%H%M%S')
            df['fecha'] = df['fecha'].dt.normalize()

            return df[['fecha', 'texto_noticia', 'source', 'url']]

        except RequestException as e:
            print(f"Error en la llamada a la API de Alpha Vantage: {e}")
            return empty_df
        except (KeyError, TypeError, ValueError, JSONDecodeError) as e:
            print(f"Error procesando la respuesta JSON de la API: {e}")
            return empty_df
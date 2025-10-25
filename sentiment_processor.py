import re
from typing import List, Dict, Any, Callable

import pandas as pd
import torch
# import nltk # No es necesario si VADER se instala por separado
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Pipeline
)

class FinancialSentimentProcessor:
    """
    Procesa texto de noticias financieras para calcular y agregar
    puntuaciones de sentimiento de VADER y FinBERT.
    """

    def __init__(self, finbert_model_name: str = "ProsusAI/finbert"):
        self.vader_analyzer = SentimentIntensityAnalyzer()

        # Detección de CUDA
        self.device: int = 0 if torch.cuda.is_available() else -1
        
        tokenizer = AutoTokenizer.from_pretrained(finbert_model_name)
        model = AutoModelForSequenceClassification.from_pretrained(finbert_model_name)

        self.finbert_pipeline: Pipeline = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            device=self.device
        )

    def _clean_text(self, text: str) -> str:
        """limpia el texto de las noticias."""
        if not isinstance(text, str):
            return ""
        
        text = text.lower()
        text = re.sub(r'http\S+', '', text)  # Eliminar URLs
        text = re.sub(r'\$\w+', '', text)  # Eliminar tickers
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r'[^a-z0-9\s.,]', '', text) # Mantenemos la sugerencia original, puedes añadir !? si quieres probar VADER con más puntuación
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _map_finbert_score(self, result: Dict[str, Any]) -> float:
        """Mapea la salida de FinBERT a una puntuación de polaridad (-1 a 1)."""
        label = result['label']
        score = result['score']

        if label == 'positive':
            return score
        elif label == 'negative':
            return -score
        else:  # 'neutral'
            return 0.0

    def calculate_sentiments(
        self,
        df: pd.DataFrame,
        text_col: str = 'texto_noticia'
    ) -> pd.DataFrame:
        
        if text_col not in df.columns:
            raise ValueError(f"La columna '{text_col}' no se encuentra en el DataFrame.")
        
        if df.empty:
            df['vader_score'] = pd.Series(dtype='float64')
            df['finbert_score'] = pd.Series(dtype='float64')
            return df

        df_copy = df.copy()

        cleaned_texts: pd.Series = df_copy[text_col].apply(self._clean_text)

        # Cálculo de VADER 
        df_copy['vader_score'] = cleaned_texts.apply(
            lambda x: self.vader_analyzer.polarity_scores(x)['compound']
        )

        # Cálculo de FinBERT (Procesamiento por lotes manual) 
        text_list: List[str] = cleaned_texts.tolist()
        finbert_results_list: List[Dict[str, Any]] = []
        batch_size = 32 # Batch size más pequeño para prevenir OOM

        print(f"Procesando {len(text_list)} textos con FinBERT en lotes de {batch_size}...")

        for i in range(0, len(text_list), batch_size):
            batch = text_list[i : i + batch_size]
            
            try:
                finbert_results = self.finbert_pipeline(
                    batch,
                    truncation=True
                )
                finbert_results_list.extend(finbert_results)
                
            except RuntimeError as e: # Captura OOM (Out Of Memory)
                print(f"ADVERTENCIA: Error Runtime en FinBERT (lote {i}). Asignando 0.0 a este lote. Error: {e}")
                neutral_result = {'label': 'neutral', 'score': 0.0}
                finbert_results_list.extend([neutral_result] * len(batch))
                
            except Exception as e:
                print(f"ADVERTENCIA: Error inesperado en FinBERT (lote {i}). Asignando 0.0 a este lote. Error: {e}")
                neutral_result = {'label': 'neutral', 'score': 0.0}
                finbert_results_list.extend([neutral_result] * len(batch))

        df_copy['finbert_score'] = [self._map_finbert_score(res) for res in finbert_results_list]
        print("Procesamiento FinBERT completado.")
        # --- FIN DEL CAMBIO ---

        return df_copy

    def _safe_ratio(self, x: pd.Series, condition: Callable[[pd.Series], pd.Series]) -> float:
        """Helper para calcular ratios"""
        count = x.count()
        if count == 0:
            return 0.0
        return condition(x).sum() / count

    def aggregate_features(
        self,
        df_with_scores: pd.DataFrame,
        date_col: str = 'fecha'
    ) -> pd.DataFrame:
        
        if 'finbert_score' not in df_with_scores.columns or \
           'vader_score' not in df_with_scores.columns:
            raise ValueError(
                "Columnas 'finbert_score' o 'vader_score' no encontradas."
            )
        
        if df_with_scores.empty:
            return pd.DataFrame()

        # Agregaciones de VADER 
        grouped_vader = df_with_scores.groupby(date_col)['vader_score']
        aggs_vader = {
            'vader_mean': grouped_vader.mean(),
            'vader_std': grouped_vader.std(),
            'vader_news_count': grouped_vader.count(),
            'vader_positive_ratio': grouped_vader.apply(self._safe_ratio, condition=lambda x: x > 0.05),
            'vader_negative_ratio': grouped_vader.apply(self._safe_ratio, condition=lambda x: x < -0.05),
            'vader_net_sentiment': grouped_vader.sum()
        }
        features_vader_df = pd.DataFrame(aggs_vader)

        # Agregaciones de FinBERT 
        grouped_finbert = df_with_scores.groupby(date_col)['finbert_score']
        aggs_finbert = {
            'finbert_mean': grouped_finbert.mean(),
            'finbert_std': grouped_finbert.std(),
            'finbert_news_count': grouped_finbert.count(),
            'finbert_positive_ratio': grouped_finbert.apply(self._safe_ratio, condition=lambda x: x > 0),
            'finbert_negative_ratio': grouped_finbert.apply(self._safe_ratio, condition=lambda x: x < 0),
            'finbert_net_sentiment': grouped_finbert.sum()
        }
        features_finbert_df = pd.DataFrame(aggs_finbert)
        
        features_df = pd.concat([features_vader_df, features_finbert_df], axis=1)

        features_df = features_df.fillna(0)

        features_df.index = pd.to_datetime(features_df.index)
        features_df = features_df.sort_index()

        return features_df
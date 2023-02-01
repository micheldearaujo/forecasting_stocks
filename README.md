
[![Previsão de ações](https://github.com/micheldearaujo/forecasting_stocks/actions/workflows/main.yml/badge.svg)](https://github.com/micheldearaujo/forecasting_stocks/actions/workflows/main.yml)

# Previsão de preço de ações

O objetivo deste projeto é criar um modelo prescritivo com o objetivo de informar qual é o melhor dia para comprar ou vender ações baseado em parâmetros setados pelo usuário.

O produto é baseado em um modelo de Forecasting (inicialmente XGBoost, modelos ainda em construção) que irá prevê o preço de ações (a escolha do usuário) para os próximos 10 dias úteis, e partir desses valores irá realizar recomendações de compra ou venda, baseado em estratégias (to be defined).
## Authors

- [@micheldearaujo](https://github.com/micheldearaujo/forecasting_stocks)
- [Linkedin](https://www.linkedin.com/in/michel-de-ara%C3%BAjo-947377197/)


## Installation

Para instalar o projeto localmente

```bash
  git clone https://github.com/micheldearaujo/forecasting_stocks.git
  cd forecasting_stocks
  python3 -m venv forecasting_stocks
  source forecasting_stocks/bin/activate
  make install
  make lint
```
## Usage/Examples

```python
  mkdir data
  mkdir data/raw
  mkdir data/processed
  python3 src/train_model.py
  python3 src/predict_model.py

```


import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px

# Criação do dataframe
df = pd.DataFrame({
   'date': pd.date_range(start='2022-01-01', end='2022-12-31', freq='D'),
   'Class': np.random.choice(['Forecasting', 'Historical'], size=365),
   'price': np.random.randint(1, high=1000, size=365)
})

# Opções de cor para as classes
color_options = ['red', 'blue']

# Título do app
st.title('Gráfico de preços')

# Filtro de data
start_date = st.sidebar.date_input("Escolha a data de início:", value=df['date'].min())
end_date = st.sidebar.date_input("Escolha a data de fim:", value=df['date'].max())

# Filtra o dataframe com base na data escolhida
mask = (df['date'] >= str(start_date)) & (df['date'] <= str(end_date))
filtered_df = df.loc[mask]

# Seleção de cores
color_forecasting = st.sidebar.selectbox("Escolha a cor para a classe Forecasting:", options=color_options, index=0)
color_historical = st.sidebar.selectbox("Escolha a cor para a classe Historical:", options=color_options, index=1)

# Gráfico de linhas
fig = px.line(filtered_df, x='date', y='price', color='Class', color_discrete_map={'Forecasting': color_forecasting, 'Historical': color_historical})

fig.update_layout(
    title="Preço ao longo do tempo",
    xaxis_title="Data",
    yaxis_title="Preço"
)

st.plotly_chart(fig)

# Medida de resumo para a classe Forecasting
forecasting_summary = filtered_df[filtered_df['Class'] == 'Forecasting']['price'].agg(['sum', 'mean', 'min', 'max'])
st.write("Resumo para a classe Forecasting:")
st.write(forecasting_summary)

# Opção para enviar feedback
feedback = st.text_input("Envie seu feedback:")
if feedback:
    st.write("Obrigado pelo seu feedback:", feedback.upper())

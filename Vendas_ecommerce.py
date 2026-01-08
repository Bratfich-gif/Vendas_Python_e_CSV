import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import os

print("Carregando dados...")
df = pd.read_csv('vendas_ecommerce.csv')
df = df.rename(columns={'data_venda': 'ds', 'valor_total': 'y'})
df['ds'] = pd.to_datetime(df['ds'])

print("Treinando o modelo de previsão...")
modelo = Prophet(yearly_seasonality=True, weekly_seasonality=True)
modelo.add_country_holidays(country_name='BR')
modelo.fit(df)

futuro = modelo.make_future_dataframe(periods=90)
previsao = modelo.predict(futuro)

print("Gerando gráficos...")
fig1 = modelo.plot(previsao)
plt.title("Previsão de Vendas - Próximos 90 Dias")
plt.xlabel("Data")
plt.ylabel("Valor (R$)")
plt.show()

flg2 = modelo.plot_components(previsao)
plt.show()

print("Processo concluído com sucesso!")
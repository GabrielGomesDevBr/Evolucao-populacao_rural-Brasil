#!/usr/bin/env python
# coding: utf-8

# Verificar a evolução da população rural no Brasil
# 

# In[46]:


#Importaação de  bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[47]:


#importar os dados
pd.read_csv ('populacao_urbana_rural.csv')


# In[48]:


#Renomear arquivo para facilitar operações
df = pd.read_csv ('populacao_urbana_rural.csv')


# In[49]:


#Verificando se ocorreu a mudança de nome
df


# In[50]:


#verificando quantidade de linhas e colunas
df.shape


# In[51]:


#verificando as primeiras linhas
df.head()


# In[52]:


#verificando as últimas linhas
df.tail()


# In[53]:


#verificando as estatísticas descritivas 
#desvio padrão, mínimo, 25º percentil (Q1), mediana (50º percentil ou Q2), 75º percentil (Q3) e máximo.
df.describe()


# In[64]:


#Criando Gráficos 
plt.figure(figsize=(10, 6))
plt.plot(df['ano'], df['urbana'], label='População Urbana', marker='o')
plt.plot(df['ano'], df['rural'], label='População Rural', marker='o')
plt.title('Evolução da População Urbana e Rural no Brasil')
plt.xlabel('Ano')
plt.ylabel('População')
plt.legend()
plt.show()


# In[55]:


#calculando taxa de crescimento
df['crescimento_urbano'] = df['urbana'].pct_change() * 100
df['crescimento_rural'] = df['rural'].pct_change() * 100


# In[56]:


df


# In[65]:


#Fazendo comparação percentual
df['total'] = df['urbana'] + df['rural']
df['percentual_urbana'] = (df['urbana'] / df['total']) * 100
df['percentual_rural'] = (df['rural'] / df['total']) * 100
df


# In[66]:


#Fazendo correlação 
df[['urbana', 'rural']].corr()


# In[59]:


df


# In[60]:


# Calcular a população total
df['total'] = df['urbana'] + df['rural']

# Calcular os percentuais
df['percentual_urbana'] = (df['urbana'] / df['total']) * 100
df['percentual_rural'] = (df['rural'] / df['total']) * 100

# Criar um gráfico de setor
labels = ['População Urbana', 'População Rural']
sizes = [df['percentual_urbana'].iloc[-1], df['percentual_rural'].iloc[-1]]  # Usando os dados do último ano
colors = ['lightcoral', 'lightskyblue']
explode = (0.1, 0)  # Explodir a primeira fatia (população urbana)

plt.figure(figsize=(8, 8))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.title('Distribuição Percentual da População Urbana e Rural no Último Ano')
plt.show()


# In[63]:


# Criar um gráfico de barras
plt.figure(figsize=(12, 6))
bar_width = 0.35
bar_positions_urbana = range(len(df['ano']))
bar_positions_rural = [pos + bar_width for pos in bar_positions_urbana]

plt.bar(bar_positions_urbana, df['urbana'], width=bar_width, label='População Urbana', color='lightcoral')
plt.bar(bar_positions_rural, df['rural'], width=bar_width, label='População Rural', color='lightskyblue')

# Configurar o eixo x
plt.xticks([pos + bar_width / 2 for pos in bar_positions_urbana], df['ano'], rotation=45, ha='right')

# Adicionar rótulos e título
plt.xlabel('Ano')
plt.ylabel('População')
plt.title('Evolução da População Urbana e Rural no Brasil (Gráfico de Barras)')

# Adicionar legenda
plt.legend()

# Exibir o gráfico
plt.tight_layout()
plt.show()



# In[62]:


# Criar um scatter plot
plt.figure(figsize=(12, 8))

# Plotar os pontos no scatter plot
plt.scatter(df['urbana'], df['rural'], c=df['ano'], cmap='viridis', s=100, alpha=0.8, edgecolors='w', linewidths=1)

# Adicionar rótulos e título
plt.title('Relação entre População Urbana e Rural no Brasil ao Longo dos Anos (Scatter Plot)')
plt.xlabel('População Urbana')
plt.ylabel('População Rural')

# Adicionar barra de cores (indicando o ano)
cbar = plt.colorbar()
cbar.set_label('Ano')

# Exibir o scatter plot
plt.show()


# In[68]:


#previsão para os próximos 10 anos
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Supondo que você tenha os dados em um DataFrame chamado 'df'
# Certifique-se de que 'ano', 'urbana' e 'rural' são as colunas relevantes no DataFrame

# Criar uma coluna com o total da população
df['total'] = df['urbana'] + df['rural']

# Dividir os dados em treino e teste
X = df[['ano']]
y = df['total']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar e treinar o modelo de regressão linear
model = LinearRegression()
model.fit(X_train, y_train)

# Fazer previsões para os próximos 10 anos
anos_futuros = np.arange(df['ano'].max() + 1, df['ano'].max() + 11).reshape(-1, 1)
previsoes = model.predict(anos_futuros)

# Avaliar o desempenho do modelo nos dados de teste
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Plotar os resultados
plt.figure(figsize=(12, 6))
plt.scatter(X, y, color='blue', label='Dados Históricos')
plt.plot(anos_futuros, previsoes, color='red', linestyle='dashed', linewidth=2, label='Previsões Futuras')
plt.title('Previsão da População Futura no Brasil')
plt.xlabel('Ano')
plt.ylabel('População Total')
plt.legend()
plt.show()


# In[ ]:





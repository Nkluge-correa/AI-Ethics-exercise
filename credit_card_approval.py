# ----------------------------------------------------------------------------------------#
# Bibliotecas
# ----------------------------------------------------------------------------------------#
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.core.display import display, HTML
import base64
import array as arr
import random
from facets_overview.feature_statistics_generator import FeatureStatisticsGenerator
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import ipywidgets #from google.colab import widgets

# ----------------------------------------------------------------------------------------#
# Load Data
# ----------------------------------------------------------------------------------------#
data = pd.read_csv(r'data\data.csv', header = 0)
data.columns =['Gender', 'Age', 'Debt', 'Married', 'Bank Client', 'Education',
                  'Race', 'Years Employed', 'Prior Default', 'Employed', 'Credit',
                  "Driver's License", 'Citizenship', 'Postal code', 'Income', 'Approval Status']

# ----------------------------------------------------------------------------------------#
# Show Data
# ----------------------------------------------------------------------------------------#
print(
    data,
    data.describe(include = 'all'),
    data.info()
)

# ----------------------------------------------------------------------------------------#
# Pré-processamento
# ----------------------------------------------------------------------------------------#

data = data.replace('?', np.nan) # Substituir os '?'s por NaN
data.fillna(data.mean(), inplace=True) # Imputar os valores em falta com valores médios
for col in data.columns:
    if data[col].dtypes == 'object':        
        data = data.fillna(data[col].value_counts().index[0]) # Imputar com o valor mais freqüente em colunas do tipo 'object'

# ----------------------------------------------------------------------------------------#
# Vizualizando dados com Plotly
# ----------------------------------------------------------------------------------------#
fig = make_subplots(rows=9, cols=1, subplot_titles=('<b> Gender', 
                                                    '<b> Married',
                                                    '<b> Bank Client', 
                                                    '<b> Education',
                                                    '<b> Race', 
                                                    '<b> Prior Default',
                                                    '<b> Employed',
                                                    '<b> Citizenship',
                                                    '<b> Approval Status'))

fig.add_trace(
    go.Histogram(
        x=data['Gender'], 
        nbinsx=len(data['Gender'].unique()),
        showlegend= False
        ), row=1, col=1)

fig.add_trace(
    go.Histogram(
        x=data['Married'], 
        nbinsx=len(data['Married'].unique()), 
        showlegend= False
        ), row=2, col=1)

fig.add_trace(
    go.Histogram(
        x=data['Bank Client'], 
        nbinsx=len(data['Bank Client'].unique()), 
        showlegend= False
        ), row=3, col=1)

fig.add_trace(
    go.Histogram(
        x=data['Education'], 
        nbinsx=len(data['Education'].unique()), 
        showlegend= False
        ), row=4, col=1)

fig.add_trace(
    go.Histogram(
        x=data['Race'], 
        nbinsx=len(data['Race'].unique()), 
        showlegend= False
        ), row=5, col=1)

fig.add_trace(
    go.Histogram(
        x=data['Prior Default'], 
        nbinsx=len(data['Prior Default'].unique()), 
        showlegend= False
        ), row=6, col=1)

fig.add_trace(
    go.Histogram(
        x=data['Employed'], 
        nbinsx=len(data['Employed'].unique()), 
        showlegend= False
        ), row=7, col=1)

fig.add_trace(
    go.Histogram(
        x=data['Citizenship'], 
        nbinsx=len(data['Citizenship'].unique()), 
        showlegend= False
        ), row=8, col=1)

fig.add_trace(
    go.Histogram(
        x=data['Approval Status'], 
        nbinsx=len(data['Approval Status'].unique()), 
        showlegend= False
        ), row=9, col=1)    

fig.update_layout(title={'text': '<b> Distribuição de Features',
        'y':.995, 'x':.5, 'xanchor': 'center' ,'yanchor': 'top'}, font_size= 18)
fig.update_layout(hovermode='x unified')
fig.update_layout(height=2650, width=1550, showlegend=False, autosize=False)
fig.show()

# ----------------------------------------------------------------------------------------#
# Vizualizando dados com Facets
# ----------------------------------------------------------------------------------------#
fsg = FeatureStatisticsGenerator()
dataframes = [
    {'table': data, 'name': 'trainData'}]
censusProto = fsg.ProtoFromDataFrames(dataframes)
protostr = base64.b64encode(censusProto.SerializeToString()).decode('utf-8')


HTML_TEMPLATE = '''<script src='https://cdnjs.cloudflare.com/ajax/libs/webcomponentsjs/1.3.3/webcomponents-lite.js'></script>
        <link rel='import' href='https://raw.githubusercontent.com/PAIR-code/facets/1.0.0/facets-dist/facets-jupyter.html'>
        <facets-overview id='elem'></facets-overview>
        <script>
          document.querySelector('#elem').protoInput = '{protostr}';
        </script>'''
html = HTML_TEMPLATE.format(protostr=protostr)
display(HTML(html))


# ----------------------------------------------------------------------------------------#
# Vizualizando dados com Facets Dive - only on Colab
# ----------------------------------------------------------------------------------------#
#SAMPLE_SIZE = 100
  
#data_dive = data.sample(SAMPLE_SIZE).to_json(orient='records')

#HTML_TEMPLATE = '''<script src='https://cdnjs.cloudflare.com/ajax/libs/webcomponentsjs/1.3.3/webcomponents-lite.js'></script>
#        <link rel='import' href='https://raw.githubusercontent.com/PAIR-code/facets/1.0.0/facets-dist/facets-jupyter.html'>
#        <facets-dive id='elem' height='600'></facets-dive>
#        <script>
#          var data = {jsonstr};
#          document.querySelector('#elem').data = data;
#        </script>'''
#html = HTML_TEMPLATE.format(jsonstr=data_dive)
#display(HTML(html))


# ----------------------------------------------------------------------------------------#
# Pré-processamento (Converter os dados não-numéricos em numéricos)
# ----------------------------------------------------------------------------------------#
encoder = LabelEncoder()

for col in data.columns:
    if data[col].dtype=='object':
      data[col]=encoder.fit_transform(data[col])

# ----------------------------------------------------------------------------------------#
# Split de treinamento/testagem
# ----------------------------------------------------------------------------------------#

data = data.values
X,y = data[:,0:15] , data[:,15]
X_train, X_test, y_train, y_test = train_test_split(
                                                    X,
                                                    y,
                                                    test_size=0.33,
                                                    random_state=42
                                                    )

# ----------------------------------------------------------------------------------------#
# Normalização
# ----------------------------------------------------------------------------------------#

scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX_train = scaler.fit_transform(X_train)
rescaledX_test = scaler.fit_transform(X_test)

# ----------------------------------------------------------------------------------------#
# Modelo (RL)
# ----------------------------------------------------------------------------------------#

model = LogisticRegression(penalty='l2',verbose=1)
model.fit(rescaledX_train, y_train)

# ----------------------------------------------------------------------------------------#
#Coeficientes de Correlação 
# ----------------------------------------------------------------------------------------#
#Coeficientes de Correlação medem a associação linear entre variáveis. 
#Podemos interpretar tais valores da seguinte forma:

#+1 - Correlação positiva completa;
#+0,8 - Forte correlação positiva;
#+0,6 - Correlação positiva moderada;
#0 - Sem qualquer correlação;
#-0,6 - Correlação negativa moderada;
#-0,8 - Forte correlação negativa;
#-1 - Correlação negativa completa.


features = ['Gender', 'Age', 'Debt', 'Married', 'Bank Client', 'Education',
                  'Race', 'Years Employed', 'Prior Default', 'Employed', 'Credit',
                  "Driver's License", 'Citizenship', 'Postal code', 'Income']
array = []
for i in range(0, 15):                  
    A,B = data[:,i] , data[:,15]
    coef = np.corrcoef(A,B)
    coef = pd.DataFrame(coef, index= [features[i], 'Approval Status'],
                      columns= [features[i], 'Approval Status'])
    array.append(coef[features[i]][1])
coef = pd.DataFrame(array, columns=['Correlation Coefficients'],
    index=features)
print(coef)

# ----------------------------------------------------------------------------------------#
#Testando a performance do Modelo (Matriz de COnfusão)
# ----------------------------------------------------------------------------------------#
y_pred = model.predict(rescaledX_test)
df = confusion_matrix(y_test, y_pred)
confusão = pd.DataFrame(df, index= ['True Class (Negative)', 
                                    'True Class (Positive) '],
                      columns= ['Predicted Class (Negative)', 
                                'Predicted Class (Positive)'])
print('\033[1m' + 'Performance (accuracy) do modelo: ',
    '\n', 
    model.score(rescaledX_test, y_test),
    '\n',
    confusão)

# ----------------------------------------------------------------------------------------#
#Exemplo Adversarial (Casos Extremos)
# ----------------------------------------------------------------------------------------#
a = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
b = np.array([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])
c = np.array([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])

#True Sample Example
#array([0.        , 0.84393064, 0.02982143, 0.5       , 0.        ,
#       0.        , 0.88888889, 0.01754386, 0.        , 0.        ,
#       0.        , 1.        , 0.        , 0.49112426, 0.00228963])

print(
    'Status de Aprovação:', model.predict(a), ' Matriz de Probabilidade >',
    model.predict_proba(a),
    '\n',
    'Status de Aprovação:', model.predict(b), ' Matriz de Probabilidade >',
    model.predict_proba(b),
    '\n',
    'Status de Aprovação:', model.predict(c), ' Matriz de Probabilidade >',
    model.predict_proba(c),
)


# ----------------------------------------------------------------------------------------#
#Coeficientes do Modelo
# ----------------------------------------------------------------------------------------#
coefs = pd.DataFrame(
    model.coef_,
    columns=features,
    index=['Coefficients'])
coefs = coefs.transpose()

fig = go.Figure(go.Bar(
            x=coefs['Coefficients'],
            y=features,
            orientation='h',
                marker=dict(
        color='rgba(0, 213, 255, 0.6)',
        line=dict(color='rgba(0, 213, 255, 1.0)', width=3)
        )))
fig.update_xaxes(range=[model.coef_.min() + (model.coef_.min() * 0.1), model.coef_.max() + (model.coef_.max() * 0.1)])
fig.update_layout(
    xaxis = dict(
        tickmode = 'linear',
        tick0 = 0,
        dtick = 0.5
    ),
    title='Coeficientes do Modelo-RL',
)
fig.show()

# ----------------------------------------------------------------------------------------#
#Coeficientes do Modelo (Normalizados)
# ----------------------------------------------------------------------------------------#
#Multiplicar os coeficientes pelo desvio padrão da característica 
#relacionada reduz todos os coeficientes para a mesma unidade de medida. 
#Quanto maior a variação de uma característica, maior o peso do coeficiente 
#correspondente na saída (todos os outros valores se mantendo igual). 
#O gráfico abaixo nos fala das dependências entre uma característica 
#específica e o alvo quando todas as outras características permanecem 
#constantes, i.e., as dependências condicionais.

coeffs = pd.DataFrame(model.coef_ * rescaledX_train.std(axis=0),
    columns=features,
    index=['Coeficientes Normalizados'])
coeffs = coeffs.transpose()

fig = go.Figure(go.Bar(
            x=coeffs['Coeficientes Normalizados'],
            y=features,
            orientation='h',
                marker=dict(
        color='rgba(0, 213, 255, 0.6)',
        line=dict(color='rgba(0, 213, 255, 1.0)', width=3)
        )))
fig.update_xaxes(range=[model.coef_.min() + (model.coef_.min() * 0.1), model.coef_.max() + (model.coef_.max() * 0.1)])
fig.update_layout(
    xaxis = dict(
        tickmode = 'linear',
        tick0 = 0,
        dtick = 0.5
    ),
    title='Coeficientes (Normalizados por STD) do Modelo-RL',
)
fig.show()


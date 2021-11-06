Carta de Modelo – Aprovação de Cartão de Crédito

Detalhes do Modelo

1.	Modelo desenvolvido por Nicholas Kluge, pesquisador da Pontifícia Universidade Católica do Rio Grande do Sul (PUCRS), em outubro de 2021; 
2.	Trata-se de um modelo de Regressão Logística para classificação binária, versão 0.1. Este modelo foi treinado para classificar solicitações de cartão de crédito como “Reprovadas” ou “Aprovadas”; 
3.	Este modelo foi treinado apenas por motivações acadêmicas, e ele não segue nenhum tipo de restrição de equidade/justiça. Ele não foi criado para ser implementado em aplicações reais;
4.	O conjunto de dados utilizado é o Credit Approval Data Set da UCI Machine Learning Repository. Disponível em: http://archive.ics.uci.edu/ml/datasets/credit+approval; 
5.	O código para este modelo pode ser encontrado em: https://github.com/Nkluge-correa/AI-Ethics-exercise; 
6.	Licença: MIT License;
7.	Contato: nicholas.correa@acad.pucrs.br. 

Uso Pretendido

1.	O uso pretendido deste modelo, e o código compartilhado, é apresentar ao desenvolvedor algumas ferramentas para se explorar um conjunto de dados, e avaliar possíveis implicações éticas e falhas de segurança de um modelo treinado por aprendizagem de máquina. Este modelo, e código, não foram criados para serem utilizados em aplicações reais. Contudo, as ferramentas utilizadas podem sim ser utilizadas para avaliações éticas de modelos treinados por aprendizagem de máquina;
2.	Este modelo foi desenvolvido para o público acadêmico, desenvolvedores e praticantes de aprendizagem de máquina interessados em aprender como desenvolver modelos “justos”;
3.	Como um experimento acadêmico, a única utilização para este modelo é a classificação de solicitações de cartão de crédito de amostras retiradas do Credit Approval Data Set Este modelo não deve ser usado para, e.g., classificação de score de crédito, inferência de score de crédito, ou qualquer outro tipo de tarefa diferente do seu uso primário pretendido.

Fatores

1.	As características utilizadas para a tarefa de classificar o Status de Aprovação de um solicitante de cartão de crédito são: “Gênero”, “Idade”, “Dívida”, “Casado”, “Cliente do Banco”, “Nível de Educação”, “Raça”, “Anos de Emprego”, “Inadimplência Prévia”, “Empregado”, “Crédito”, “Carteira de Motorista”, “Cidadão”, “Código Postal”, “Renda”. Atributos como “Gênero” e “Raça” são considerados como atributos sensíveis;
2.	Os dados utilizados para treinamento não possuem uma distribuição uniforme entre os subgrupos de cada característica. Existe um forte enviesamento, para certos tipos de subgrupos, como gêneros e raças específicas.

Métricas

1.	A métrica de performance utilizada foi acurácia (nº total de classificações corretas por total de classificações realizadas), 85% de acerto durante a faze de teste;
2.	O modelo possui uma tendência maior para classificar pessoas que deveriam ser aprovadas como reprovados (Falsos Negativos = 11%), do que aprovar pessoas que deveriam ser reprovadas (Falsos Positivos = 0.2%); 
3.	Sugestão: reprovações devem ser melhor investigadas/analisadas; 
4.	Dados de treinamento e testagem foram divididos do conjunto de dados fornecidos pela UCI Machine Learning Repository (i.e., Credit Approval Data Set).
5.	Este conjunto de dados foi escolhido por sua disponibilidade pública.
6.	Amostras com valores ausentes (i.e., “”?” ou “NaN”) tiveram tais valores substituídos pelo valor médio de sua característica específica.


Considerações Éticas

1.	Dada a distribuição enviesada dos dados de treinamento, o modelo pode se comportar de forma ineficiente quando lidando com amostras pouco vistas;
2.	O modelo utiliza de dados sensíveis (i.e., Raça e Gênero);
3.	Recomenda-se que para aplicações reais, atributos sensíveis (e.g., raça e gênero) e atributos contendo valores “anormais” (e.g., renda) não sejam utilizados para classificação;
4.	De acordo com os coeficientes de correlação, e coeficientes aprendidos pelo modelo, atributos sensíveis não interferem na classificação do modelo;
5.	Os atributos mais correlacionados com o Status de Aprovação do solicitante são: “Inadimplência Prévia”, “Dívida”, “Empregado” e “Crédito”.

Detalhes e Recomendações

1.	Não foi realizada uma análise da performance do modelo entre diferentes subgrupos de cada característica. Uma análise mais aprofundada pode revelar que o modelo viola critérios de equidade, como, por exemplo, paridade preditiva;
2.	Os dados utilizados para este exemplo não refletem o contexto social e histórico de um lugar como, por exemplo, Brasil. Eles refletem o contexto social e histórico Norte-Americano. Assim, não se recomenda utilizá-lo para desenvolvimento de aplicações fora deste domínio específico.

Análise Quantitativa
 
<p align="center">
<img alt="pattern demo" src="https://gdurl.com/EO0u">
</p>



 


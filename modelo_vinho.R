# Projeto Machine Learning - Wine Quality UCI

#### 1 - Introdução ####

# Objetivo do projeto

# Criar um modelo preditivo capaz de prever a qualidade de um determinado vinho.

# Sobre o dataset

# O dados foram obtidos a partir de testes feitos em amostras, como o pH por exemplo,
# e com base na avaliação de experts em vinho para determinar a sua qualidade. 
# Feito com uma base em média de 3 avaliações.

# Os dois conjuntos de dados estão relacionados com variantes tinto e branco do vinho português "Vinho Verde". 
# Para mais detalhes, consulte: [Web Link] ou a referência [Cortez et al., 2009]. 
# Por questões de privacidade e logística, apenas variáveis físico-químicas (entradas) 
# e sensoriais (saídas) estão disponíveis 
# (por exemplo, não há dados sobre tipos de uva, marca de vinho, preço de venda do vinho, etc.).

# Números de registros

# Vinho tinto - 1.599 registros
# Vinho branco - 4.898 registros

# Número de atributos

# 11 + variável target (output)

# Dicionário de Dados

# 1 - fixed acidity(g/dm^3): Ácido tartático, confere gosto amargo e encontrado em sendimentos de vinhos
# 2 - volatile acidity(g/dm^3):  O ácido acético é um ácido orgânico que se forma durante a fermentação alcoólica de forma natural. Característica essencial dos vinhos, contribui decisivamente para o seu sabor, frescura, equilíbrio e capacidade de conservação
# 3 - citric acid(g/dm^3):  A acidez de um vinho é essencial para identificar o aroma e sabor, contribuindo para sua conservação e envelhecimento
# 4 - residual sugar(g/dm^3):  Sobra de açucar resultante da fermentação das uvas
# 5 - chlorides(g/dm^3): Atua no gosto do vinho, teor muito alto resulta em um gosto mais salgado
# 6 - free sulfur dioxide (g/dm^3): Atua na qualidade do vinho para um melhor processo de fermentação aumentando a qualidade geral e longeividade do vinho
# 7 - total sulfur dioxide (g/dm^3): Atua na qualidade do vinho para um melhor processo de fermentação aumentando a qualidade geral e longeividade do vinho
# 8 - density (g/dm^3): A densidade do vinho está relacionada principalmente ao seu teor alcoólico e de açúcares residuais
# 9 - pH: Os níveis estão ligados ao estilo e qualidade dos vinhos. Geralmente um vinho com níveis de pH mais baixos terá maior longevidade.
# 10 - sulphates (g/dm3): Utilizado como conservante para o vinho.
# 11 - alcohol (% volume): Volume de álccol presente na bebida

# Variável target:
# 12 - quality (0 a 10):

# Fonte dos dados

# Paulo Cortez, University of Minho, Guimarães, Portugal, http://www3.dsi.uminho.pt/pcortez
# A. Cerdeira, F. Almeida, T. Matos and J. Reis, Viticulture Commission of the Vinho Verde Region(CVRVV), Porto, Portugal
# @2009

#### 2 - Carga dos dados ####

# Definindo diretório de trabalho
setwd('C:/FCD/R/UCI/wine_quality')
getwd()

# Carga de Pacotes
library(ggplot2)
library(grid)
library(dplyr)
library(corrplot)
library(caret)
library(gridExtra)
library(nnet)
library(rmarkdown)


# Carrega dados
wine_red <- read.csv('winequality-red.csv', sep = ";")
wine_white <- read.csv('winequality-white.csv', sep = ";")

# Criar coluna identificando o tipo de vinho
wine_red['color'] = 1 # Red Wine
wine_white['color'] = 0 # White Wine

# Único dataset para os dois vinhos
df_wine <- rbind(wine_red, wine_white)

View(df_wine)

#### 3 - Análise Exploratória ####

# Resumo estatístico inicial
summary(df_wine)

# Na análise preliminar, vemos que a média e mediana estão próximas em poucos
# atributos (ex.: pH, density e alcohol). Nas outras variáveis, temos uma diferença
# considerável. Isso indica que os dados podem precisar de remoção de outliers.

# Pelo dicionário de dados, temos atributos numéricos de diferentes escalas.
# Por esse motivo, será preciso padronizar os dados. 

# Tipos de dados
str(df_wine)

# Atributo quality e color podem ser classificadas como categóricas.
# Detalhe para a variável quality porque a mesma é target. Isso implica no 
# tipo de modelo. Para classificação, o interessante é classificar como categoria (factor)
# Já para regressão, pode-se usar o label enconding. 


# Gráfico de barras - Wine Quality
ggplot(df_wine, aes(x = as.factor(quality))) +
  geom_bar(fill = 'cadetblue' ) +
  labs(x = 'Qualidade Vinho', y = 'Contagem', title = 'Total de Registros por Qualidade')

# Fica claro que temos a maioria dos registros com qualidade média, o que é esperado. 
# Uma classificação pode criar um modelo com tendência em 6 ou 5, isso pode ser um problema

# Análise de correlação

corrplot(cor(df_wine[,1:12]), type = 'upper')

# Multicolinearidade entre as variáveis alcohol e density.
# Além disso, volatile.acidity parece ter uma correlação  negativa com quality

# Histogramas
hist(df_wine$chlorides)

hist(df_wine$density)

hist(df_wine$pH)

# Dos histogramas feitos, fica claro que nem todas as variáveis tendem a uma normal
# Isso reforça o que foi visto com relação a média e mediana de algumas variáveis
# Será preciso uma padronização dos dados

#### 4 - Pré Processamento ####

# Normalização dos dados

df <- df_wine[,1:12] # Preserva o dataset original, descartando o tipo de vinho.

names_df <- colnames(df[,1:11])
names_df # Remove a coluna target para normalização

# Função para teste de normalidade
fun.normal <- function(df, variavel){
  for (variavel in variavel){
    a <- shapiro.test(df[1:5000, variavel])
    print(paste('O valor p é: ', a$p.value))
    
  }
return()
}

fun.normal(df, names_df)



# Função para normalização dos dados
scale.features <- function(df, variavel){
  for (variavel in variavel) {
    df[[variavel]] <- scale(df[[variavel]], center = T, scale = T)
  }
  return(df)
}

df_normal <- scale.features(df, names_df)

# Dados de treino e teste
amostra_dados <- sample(x = nrow(df_normal),
                        size = 0.8 * nrow(df_normal),
                        replace = FALSE)

# Dados de treino e teste
dados_treino <- df_normal[amostra_dados,]
dados_teste <- df_normal[-amostra_dados,]


# 5 - Modelo Preditivo
modelo_v1 <- lm( quality ~ ., data = dados_treino)
summary(modelo_v1)

# var target com fator
dados_treino2 <- dados_treino
dados_treino2$quality <- as.factor(dados_treino2$quality)

dados_teste2 <- dados_teste
dados_teste2$quality <- as.factor(dados_teste2$quality)

modelo_v2 <- multinom(quality ~ ., data = dados_treino2)
summary(modelo_v2)

previsoes <- predict(modelo_v2, dados_teste)
confusionMatrix(previsoes, dados_teste2$quality)


#Ao analisar os dois modelos, fica claro que o de regressão linear multivariada não teve um bom desempenho (com um R2 de abaixo de 30%). Já o modelo de regressão logística multivariada teve um desempenho melhor (acurácia acima de 50%). Será necessário mais testes com diferentes modelos, mas a princípio uma regressão linear não funcionou muito bem porque a variável target é uma classificação, ou seja, a nota que o especilista determinou para o tipo de vinho. Modelos de classificação tendem a ter um melhor desempenho nesse sentido. 

#Próximos passos

#* Modelo com RandomForest
#* Modelo com DecisonTree
#* Remoção de alguns outliers antes de rodar os modelos. 
#* Analise de Multicolinearidade

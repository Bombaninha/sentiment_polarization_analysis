from os import replace
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import nltk
from nltk.metrics import ConfusionMatrix
import re, string

from nltk.tokenize import TweetTokenizer

def getListOfStopWords():
    return nltk.corpus.stopwords.words('portuguese')

def getDatasetLength(dataframe):
    return dataframe.shape[0]

def getListQuantityByFeeling(dataframe):
    return dataframe.Sentimento.value_counts()

def getListQuantityByFeelingPercent(dataframe):
    return (getListQuantityByFeeling(dataframe) / getDatasetLength(dataframe)) * 100

# São palavras e termos frequentes que não tem relevância nos dados
def removeStopWords(texto):
    frases = []
    stopwords = getListOfStopWords()
    for (palavras, sentimento) in texto:
        # Criamos uma list compreheension para extrair apenas as palavras que não estão na lista_Stop
        semStop = [ p for p in palavras.split() if p not in stopwords]
        # Inserindo as frases com os Labels (sentimento) já tratadas pela Lista_Stop
        frases.append((semStop, sentimento))
    return frases

def removeStopWordsIntensidade(texto):
    frases = []
    stopwords = getListOfStopWords()
    for (palavras, intensidade) in texto:
        # Criamos uma list compreheension para extrair apenas as palavras que não estão na lista_Stop
        semStop = [ p for p in palavras.split() if p not in stopwords]
        # Inserindo as frases com os Labels (sentimento) já tratadas pela Lista_Stop
        frases.append((semStop, intensidade))
    return frases

# Remoção de sufixos e prefixos
def aplica_Stemmer(texto):
    stopwords = getListOfStopWords()
    
    # RSLPS pois é especifico da lingua portuguesa
    stemmer = nltk.stem.RSLPStemmer()
    
    frases_sem_Stemming = []
    for (index, row) in texto.iterrows():
        com_Stemming = [str(stemmer.stem(p)) for p in row['Frase'].split() if p not in stopwords]
        frases_sem_Stemming.append((com_Stemming, row['Sentimento']))
    return frases_sem_Stemming

def busca_Palavras(frases):
    todas_Palavras = []
    for (palavras, sentimento) in frases:
        todas_Palavras.extend(palavras)
    return todas_Palavras

def removeNoise(words):
    newWords = []
    replaceItems = ['?', '!', ',', '.', '"']
    
    for word in words:
        x = word.replace(replaceItems[0], '').replace(replaceItems[1], '').replace(replaceItems[2], '').replace(replaceItems[3], '').replace(replaceItems[4], '')
        newWords.append(x)
    return newWords

def getFrequencyOfWords(words):
    return nltk.FreqDist(words)

def getMostFrequencyWords(frequencies, quantity):
    return frequencies.most_common(quantity)

def getUniqueWords(frequencies):
    return frequencies.keys()

def extrator_palavras(documento):
    # Utilizado set() para associar a variavel doc com o parâmetro que esta chegando
    global palavras_unicas_treinamento
    doc = set(documento)
    caracteristicas = {}
    for palavras in palavras_unicas_treinamento:
        caracteristicas['%s' % palavras] = (palavras in doc)
    return caracteristicas

def Limpeza_dados(instancia):
    # Remove Link
    instancia = re.sub(r"http\S+", "", instancia)

    # Transforma em minúsculo
    instancia = instancia.lower()
    
    # Remove os caracteres abaixo
    items = ['.', ';', '-', ':', '(', ')', '!', '"']

    for i in items:
        instancia = instancia.replace(i, '')

    return (instancia)

def Preprocessing(instancia):
    instancia = Limpeza_dados(instancia)
    stopwords = set(nltk.corpus.stopwords.words('portuguese'))
    palavras = [i for i in instancia.split() if not i in stopwords]
    return (" ".join(palavras))

# Função para aplicar as tags de negação:
def marque_negacao(texto):
    negacoes = ['não','not']
    negacao_detectada = False
    resultado = []
    palavras = texto.split()
    for p in palavras:
        p = p.lower()
        if negacao_detectada == True:
            p = p + '_NEG'
        if p in negacoes:
            negacao_detectada = True
        resultado.append(p)
    return (" ".join(resultado))

# Lemmatization: Reduz as palavras flexionadas adequadamente, garantindo que a palavra raiz pertença ao idioma
# https://core.ac.uk/download/pdf/62918901.pdf

base_treinamento = pd.read_csv('dataset/train_dataset.csv', encoding='utf-8') 
base_treinamento.columns = ['Frase', 'Sentimento']

# Transposição do cabeçalho da tabela com alguns registros
print(base_treinamento.head().T)

# Categoriza a base de treinamento por meio do sentimento
print(base_treinamento.Sentimento.value_counts())

# Plotando o gráfico de distribuição
#base_treinamento.Sentimento.value_counts().plot(kind='bar')
#plt.show()

# Eliminando repetição
print(base_treinamento.Frase.count())
base_treinamento.drop_duplicates(['Frase'], inplace=True)
print(base_treinamento.Frase.count())

stopwordsPt = nltk.corpus.stopwords.words('portuguese')
print(np.transpose(stopwordsPt))
# lista_Stop.append('tipo')

print(pd.DataFrame(base_treinamento, columns=['Frase', 'Sentimento']).sample(10))
frases_com_Stem_treinamento = aplica_Stemmer(base_treinamento)
print(pd.DataFrame(frases_com_Stem_treinamento, columns=['Frase', 'Sentimento']).sample(10))

palavras_treinamento = busca_Palavras(frases_com_Stem_treinamento) 
print(palavras_treinamento)


print("Quantidade de palavras na base de Treinamento {}".format(pd.DataFrame(palavras_treinamento).count()))

frequencia_treinamento = getFrequencyOfWords(palavras_treinamento)
print(frequencia_treinamento.most_common(20))

palavras_unicas_treinamento = getUniqueWords(frequencia_treinamento)
print(palavras_unicas_treinamento)

base_completa_treinamento = nltk.classify.apply_features(extrator_palavras, frases_com_Stem_treinamento)

classificador = nltk.NaiveBayesClassifier.train(base_completa_treinamento)

print(classificador.labels())

print(classificador.show_most_informative_features(10))

phrase = 'estou muito feliz'

teste = phrase
testeStemming = []
stemmer = nltk.stem.RSLPStemmer()
for (palavras_treinamento) in teste.split():
    comStem = [p for p in palavras_treinamento.split()]
    testeStemming.append(str(stemmer.stem(comStem[0])))

novo = extrator_palavras(testeStemming)

distribuicao = classificador.prob_classify(novo)

for classe in distribuicao.samples():
    print('%s: %f' % (classe, distribuicao.prob(classe)))
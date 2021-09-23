from os import replace
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import nltk
from nltk.metrics import ConfusionMatrix
import re, string

# https://github.com/AleLustosa/An-lise_Sentimento_NLTK/blob/master/Analise%20de%20Sentimentos%20com%20NLTK.ipynb
# https://minerandodados.com.br/analise-de-sentimentos-utilizando-dados-do-twitter/
# leticiarmachado@gmail.com

#x = pd.DataFrame(base_treinamento)
#x.to_csv(path_or_buf='dataset/train_dataset.csv', header=['Frase', 'Sentimento'], index=False)
#nltk.download('stopwords')
#nltk.download('rslp')

def getDatasetLength(dataframe):
    return dataframe.shape[0]

def getListQuantityByFeeling(dataframe):
    return dataframe.Sentimento.value_counts()

def getListQuantityByFeelingPercent(dataframe):
    return (getListQuantityByFeeling(dataframe) / getDatasetLength(dataframe)) * 100

def removeStopWords(texto):
    frases = []
    lista_Stop = nltk.corpus.stopwords.words('portuguese')
    for (palavras, sentimento) in texto:
        # Criamos uma list compreheension para extrair apenas as palavras que não estão na lista_Stop
        semStop = [ p for p in palavras.split() if p not in lista_Stop]
        # Inserindo as frases com os Labels (sentimento) já tratadas pela Lista_Stop
        frases.append((semStop, sentimento))
    return frases

def getListOfStopWords():
    return nltk.corpus.stopwords.words('portuguese')

# Remoção de sufixos e prefixos
def aplica_Stemmer(texto):
    lista_Stop = getListOfStopWords()
    
    # RSLPS pois é especifico da lingua portuguesa
    stemmer = nltk.stem.RSLPStemmer()
    
    frases_sem_Stemming = []
    for (index, row) in texto.iterrows():
        com_Stemming = [str(stemmer.stem(p)) for p in row['Frase'].split() if p not in lista_Stop]
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

def extrator_palavras_teste(documento):
    global palavras_unicas_teste
    doc = set(documento)
    caracteristicas = {}
    for palavras in palavras_unicas_teste:
        caracteristicas['%s' % palavras] = (palavras in doc)
    return caracteristicas

base_treinamento = pd.read_csv('dataset/train_dataset.csv', encoding='utf-8') 
base_treinamento.columns = ['Frase', 'Sentimento']

base_teste = pd.read_csv('dataset/test_dataset.csv', encoding='utf-8') 
base_teste.columns = ['Frase', 'Sentimento']

stopwordsPt = nltk.corpus.stopwords.words('portuguese')
# np.transpose(stopwordsPt)
# lista_Stop.append('tipo')

frases_com_Stem_treinamento = aplica_Stemmer(base_treinamento)
frases_com_Stem_teste = aplica_Stemmer(base_teste)

palavras_treinamento = busca_Palavras(frases_com_Stem_treinamento) 
palavras_teste = busca_Palavras(frases_com_Stem_teste)

palavras_treinamento = removeNoise(palavras_treinamento)
palavras_teste = removeNoise(palavras_teste)
#print(pd.DataFrame(palavras_treinamento))

#print("Quantidade de palavras na base de Treinamento {}".format(pd.DataFrame(palavras_treinamento).count()))
#print("Quantidade de palavras na base de Teste {}".format(pd.DataFrame(palavras_teste).count()))

frequencia_treinamento = getFrequencyOfWords(palavras_treinamento)
frequencia_teste = getFrequencyOfWords(palavras_teste)

#print(frequencia_treinamento)

#print(getUniqueWords(frequencia_treinamento))

palavras_unicas_treinamento = getUniqueWords(frequencia_treinamento)
palavras_unicas_teste = getUniqueWords(frequencia_teste)

#print(frequencia_treinamento)
#print(palavras_unicas_treinamento)

base_completa_treinamento = nltk.classify.apply_features(extrator_palavras, frases_com_Stem_treinamento)
base_completa_teste = nltk.classify.apply_features(extrator_palavras, frases_com_Stem_teste)

classificador = nltk.NaiveBayesClassifier.train(base_completa_treinamento)

#print(classificador.labels())
#print(classificador.show_most_informative_features(10))
#print(nltk.classify.accuracy(classificador, base_completa_teste))

'''
erros = []
for (frase, classe) in base_completa_teste:
    #print(frase)
    #print(classe)
    resultado = classificador.classify(frase)
    if resultado != classe:
        erros.append((classe, resultado, frase))

esperado = []
previsto = []
for (frase, classe) in base_completa_teste:
    resultado = classificador.classify(frase)
    previsto.append(resultado)
    esperado.append(classe)

#matriz = ConfusionMatrix(esperado, previsto)
#print(matriz)
'''

class Response:
    def __init__(self, distribuicao):
        self.classes = []
        for classe in distribuicao.samples():
            self.classes.append((classe, distribuicao.prob(classe)))

    def __str__(self):
        string = ''
        for i in self.classes:
            string += i[0] + ': ' + str(i[1]) + '\n'
        return string

def analyze(phrase):
    global classificador
    teste = phrase
    testeStemming = []
    stemmer = nltk.stem.RSLPStemmer()
    for (palavras_treinamento) in teste.split():
        comStem = [p for p in palavras_treinamento.split()]
        testeStemming.append(str(stemmer.stem(comStem[0])))

    novo = extrator_palavras(testeStemming)

    distribuicao = classificador.prob_classify(novo)
    return Response(distribuicao)
    #for classe in distribuicao.samples():
        
    #    print('%s: %f' % (classe, distribuicao.prob(classe)))

print(analyze("amor paixão apaixonado criança felicidade"))
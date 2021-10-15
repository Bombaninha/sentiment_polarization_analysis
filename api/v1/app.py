from os import replace
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import nltk
from nltk.metrics import ConfusionMatrix
import re, string

# https://github.com/AleLustosa/An-lise_Sentimento_NLTK/blob/master/Analise%20de%20Sentimentos%20com%20NLTK.ipynb
# https://minerandodados.com.br/analise-de-sentimentos-utilizando-dados-do-twitter/

#x = pd.DataFrame(base_treinamento)
#x.to_csv(path_or_buf='dataset/train_dataset.csv', header=['Frase', 'Sentimento'], index=False)

#nltk.download('stopwords')
#nltk.download('rslp')

'''
class Dataset:

    def __init__(self, dataset_path = 'dataset', dataset_file_name = 'train_dataset.csv', encoding = 'utf-8', columns = ['Frase', 'Sentimento']):
        self.set_dataset_path(dataset_path + '/' + dataset_file_name)
        self.set_encoding(encoding)
        self.set_columns(columns)

        self.initialize_dataset()

    def get_dataset_path(self):
        return self.dataset_path

    def set_dataset_path(self, dataset_path):
        self.dataset_path = dataset_path

    def get_encoding(self):
        return self.encoding

    def set_encoding(self, encoding):
        self.encoding = encoding

    def get_columns(self):
        return self.columns

    def set_columns(self, columns):
        self.columns = columns

    def initialize_dataset(self):
        self.set_dataset()
        self.set_dataset_columns()

    def get_dataset(self):
        return self.dataset

    def set_dataset(self):
        self.dataset = pd.read_csv(self.get_dataset_path(), self.get_encoding()) 

    def get_dataset_columns(self):
        return self.dataset_columns()

    def set_dataset_columns(self):
        self.dataset.columns = self.get_columns()


teste_dataset = Dataset()
print(teste_dataset)

'''
class App:
    
    def __init__(self, stopwords, train_dataset):
        self.stopwords = stopwords
        self.train_dataset = train_dataset

    def get_stopwords(self):
        return self.stopwords

    def set_stopwords(self, stopwords):
        self.stopwords = stopwords

    def add_stopword(self, stopword):
        self.stopwords.append(stopword)

    def remove_stopwords_from_data(self, text):
        phrases = []

        for (words, sentiment) in text:
            # Criamos uma list compreheension para extrair apenas as palavras que não estão na lista_Stop
            text_without_stopwords = [ p for p in words.split() if p not in self.get_stopwords()]
            # Inserindo as frases com os Labels (sentimento) já tratadas pela Lista_Stop
            phrases.append((text_without_stopwords, sentiment))
        return phrases       

#app = App()
#app.set_stopwords(nltk.corpus.stopwords.words('portuguese'))


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
    stopwords = nltk.corpus.stopwords.words('portuguese')
    for (palavras, sentimento) in texto:
        # Criamos uma list compreheension para extrair apenas as palavras que não estão na lista_Stop
        semStop = [ p for p in palavras.split() if p not in stopwords]
        # Inserindo as frases com os Labels (sentimento) já tratadas pela Lista_Stop
        frases.append((semStop, sentimento))
    return frases

def removeStopWordsIntensidade(texto):
    frases = []
    lista_Stop = nltk.corpus.stopwords.words('portuguese')
    for (palavras, intensidade) in texto:
        # Criamos uma list compreheension para extrair apenas as palavras que não estão na lista_Stop
        semStop = [ p for p in palavras.split() if p not in lista_Stop]
        # Inserindo as frases com os Labels (sentimento) já tratadas pela Lista_Stop
        frases.append((semStop, intensidade))
    return frases

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

# Remoção de sufixos e prefixos
# É a técnica de reduzir uma palavra para o seu radical, mesmo que o radical em si não seja uma palavra válida no idioma. Removendo os sufixos e prefixos de uma palavra.
def aplica_Stemmer_intensidade(texto):
    lista_Stop = getListOfStopWords()
    
    # RSLPS pois é especifico da lingua portuguesa
    stemmer = nltk.stem.RSLPStemmer()
    
    frases_sem_Stemming = []
    for (index, row) in texto.iterrows():
        com_Stemming = [str(stemmer.stem(p)) for p in row['Frase'].split() if p not in lista_Stop]
        frases_sem_Stemming.append((com_Stemming, row['Intensidade']))
    return frases_sem_Stemming

def busca_Palavras(frases):
    todas_Palavras = []
    for (palavras, sentimento) in frases:
        todas_Palavras.extend(palavras)
    return todas_Palavras

def busca_Palavras_intensidade(frases):
    todas_Palavras = []
    for (palavras, intensidade) in frases:
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

def extrator_palavras_intensidade(documento):
    # Utilizado set() para associar a variavel doc com o parâmetro que esta chegando
    global palavras_unicas_treinamento_intensidade
    doc = set(documento)
    caracteristicas = {}
    for palavras in palavras_unicas_treinamento_intensidade:
        caracteristicas['%s' % palavras] = (palavras in doc)
    return caracteristicas

def extrator_palavras_teste(documento):
    global palavras_unicas_teste
    doc = set(documento)
    caracteristicas = {}
    for palavras in palavras_unicas_teste:
        caracteristicas['%s' % palavras] = (palavras in doc)
    return caracteristicas

def Limpeza_dados(instancia):
    # remove links, pontos, virgulas,ponto e virgulas dos tweets
    instancia = re.sub(r"http\S+", "", instancia).lower().replace('.','').replace(';','').replace('-','').replace(':','').replace(')','')
    return (instancia)

# Lemmatization: Reduz as palavras flexionadas adequadamente, garantindo que a palavra raiz pertença ao idioma
# https://core.ac.uk/download/pdf/62918901.pdf

base_treinamento = pd.read_csv('dataset/train_dataset.csv', encoding='utf-8') 
base_treinamento.columns = ['Frase', 'Sentimento']

# Transposição do cabeçalho da tabela com alguns registros
print(base_treinamento.head().T)

# Categoriza a base de treinamento por meio do sentimento
print(base_treinamento.Sentimento.value_counts())

# Plotando o gráfico de distribuição
base_treinamento.Sentimento.value_counts().plot(kind='bar')
#plt.show()

# Eliminando repetição
print(base_treinamento.Frase.count())
base_treinamento.drop_duplicates(['Frase'], inplace=True)
print(base_treinamento.Frase.count())

frase = 'O Lucas é um bom companheiro, ninguém pode negar ! :) :P :B'
from nltk.tokenize import TweetTokenizer
tweet_tokenizer = TweetTokenizer() 
print(tweet_tokenizer.tokenize(frase))

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

print(marque_negacao("Eu gosto do novo dono do bar"))
print(marque_negacao("Eu não gosto do novo dono do bar"))
base_teste = pd.read_csv('dataset/test_dataset.csv', encoding='utf-8') 
base_teste.columns = ['Frase', 'Sentimento']

base_treinamento_intensidade = pd.read_csv('dataset/train_dataset_intensity.csv', encoding='utf-8') 
base_treinamento_intensidade.columns = ['Frase', 'Intensidade']

stopwordsPt = nltk.corpus.stopwords.words('portuguese')
# np.transpose(stopwordsPt)
# lista_Stop.append('tipo')

frases_com_Stem_treinamento = aplica_Stemmer(base_treinamento)
frases_com_Stem_teste = aplica_Stemmer(base_teste)
frases_com_Stem_treinamento_intensidade = aplica_Stemmer_intensidade(base_treinamento_intensidade)

palavras_treinamento = busca_Palavras(frases_com_Stem_treinamento) 
palavras_teste = busca_Palavras(frases_com_Stem_teste)
palavras_treinamento_intensidade = busca_Palavras_intensidade(frases_com_Stem_treinamento_intensidade)

palavras_treinamento = removeNoise(palavras_treinamento)
palavras_teste = removeNoise(palavras_teste)
palavras_treinamento_intensidade = removeNoise(palavras_treinamento_intensidade)
#print(pd.DataFrame(palavras_treinamento))

#print("Quantidade de palavras na base de Treinamento {}".format(pd.DataFrame(palavras_treinamento).count()))
#print("Quantidade de palavras na base de Teste {}".format(pd.DataFrame(palavras_teste).count()))

frequencia_treinamento = getFrequencyOfWords(palavras_treinamento)
frequencia_teste = getFrequencyOfWords(palavras_teste)
frequencia_treinamento_intensidade = getFrequencyOfWords(palavras_treinamento_intensidade)

#print(frequencia_treinamento)

#print(getUniqueWords(frequencia_treinamento))

palavras_unicas_treinamento = getUniqueWords(frequencia_treinamento)
palavras_unicas_teste = getUniqueWords(frequencia_teste)
palavras_unicas_treinamento_intensidade = getUniqueWords(frequencia_treinamento_intensidade)

#print(frequencia_treinamento)
#print(palavras_unicas_treinamento)

base_completa_treinamento = nltk.classify.apply_features(extrator_palavras, frases_com_Stem_treinamento)
base_completa_teste = nltk.classify.apply_features(extrator_palavras, frases_com_Stem_teste)
base_completa_treinamento_intensidade = nltk.classify.apply_features(extrator_palavras_intensidade, frases_com_Stem_treinamento_intensidade)

classificador = nltk.NaiveBayesClassifier.train(base_completa_treinamento)
classificador_intensidade = nltk.NaiveBayesClassifier.train(base_completa_treinamento_intensidade)
#print(classificador_intensidade.labels())

#print(classificador_intensidade.show_most_informative_features(10))

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
    global classificador_intensidade
    teste = phrase
    testeStemming = []
    stemmer = nltk.stem.RSLPStemmer()
    for (palavras_treinamento) in teste.split():
        comStem = [p for p in palavras_treinamento.split()]
        testeStemming.append(str(stemmer.stem(comStem[0])))

    novo = extrator_palavras(testeStemming)

    distribuicao = classificador.prob_classify(novo)
    distribuicao_intensidade = classificador_intensidade.prob_classify(novo)

    #print(distribuicao)
    #print(distribuicao_intensidade)
    #for classe in distribuicao.samples():
        #print('%s: %f' % (classe, distribuicao.prob(classe)))

    print("\n")

    #for classe in distribuicao_intensidade.samples():
        #print('%s: %f' % (classe, distribuicao_intensidade.prob(classe)))
    #return Response(distribuicao)
    #for classe in distribuicao.samples():
    #    print('%s: %f' % (classe, distribuicao.prob(classe)))

analyze("estou muito feliz")

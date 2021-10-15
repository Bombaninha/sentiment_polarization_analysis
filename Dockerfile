FROM ubuntu:latest

# Instala nossas dependencias
RUN apt-get update && apt-get install -y python pip

# Seta nosso path
ENV INSTALL_PATH /emotion_detect_api

# Cria nosso diretório
RUN mkdir -p $INSTALL_PATH

# Seta o nosso path como o diretório principal
WORKDIR $INSTALL_PATH

RUN pip install flask pandas matplotlib nltk flask_restful

# Download de stopwords
RUN nltk.download('stopwords')

# Download do stem da lingua portuguesa
RUN nltk.download('rslp')

# Seta as permissões da pasta
RUN chmod -R 777 $INSTALL_PATH

# Copia nosso código para dentro do container
COPY . .
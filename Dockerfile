FROM continuumio/anaconda3:4.4.0
MAINTAINER Joseph Oladokun, https://github.com/Godskid89
EXPOSE 8000
RUN pip install -r requirements.txt
# RUN python -m nltk.downloader averaged_tagger_perceptron
# RUN /usr/local/nlp_app/nlp-api.py
WORKDIR /usr/local/nlp_app
CMD python nlp-api.py
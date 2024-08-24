# Use the jupyter datascience-notebook
FROM jupyter/datascience-notebook

RUN pip install plotly nltk
RUN pip install pytest, ipytest
RUN jupyter labextension install jupyterlab-plotly

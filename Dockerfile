# Use the jupyter datascience-notebook
FROM jupyter/datascience-notebook

RUN pip install plotly
RUN jupyter labextension install jupyterlab-plotly

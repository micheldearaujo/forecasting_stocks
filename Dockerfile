FROM python:3.8.16

CMD mkdir /stock_forecaster
COPY . /stock_forecaster

WORKDIR /stock_forecaster

EXPOSE 8080

RUN pip3 install -r requirements.txt

# #, "src/models/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

#RUN streamlit run src/models/app.py --server.port 8080
ENTRYPOINT ["streamlit", "run"]
CMD ["app.py", "--server.port=8080"]
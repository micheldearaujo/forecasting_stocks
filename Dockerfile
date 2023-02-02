FROM python:3.8.16

CMD mkdir /stock_forecaster
COPY . /stock_forecaster

WORKDIR /stock_forecaster

EXPOSE 8080

RUN pip3 install -r requirements.txt

RUN streamlit run src/models/predict_model.py

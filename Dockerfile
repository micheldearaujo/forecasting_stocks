FROM python:3.8.16

CMD mkdir /stock_forecaster
COPY . /stock_forecaster

WORKDIR /stock_forecaster

EXPOSE 8080

RUN pip3 install -r requirements.txt

RUN python3 -m streamlit run src/models/app.py --server.port 8080

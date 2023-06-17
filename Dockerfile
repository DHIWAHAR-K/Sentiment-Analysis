FROM python:3.10.11
COPY ./app
WORKDIR /app
RUN pip install -r requirements.txt
EXPORSE $PORT
CMD gunicorn --workers=4 ---blind 0.0.0.0:$PORT app:app

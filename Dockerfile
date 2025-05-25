FROM python:3.10

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y libgl1

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]
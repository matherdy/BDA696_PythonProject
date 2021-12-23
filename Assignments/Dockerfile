
FROM python:3.8.2

WORKDIR /app



RUN apt-get update && apt-get -y install default-mysql-client \
    build-essential \
     python3 \
     python3-pip \
     python3-dev

COPY requirements.txt /app/requirements.txt

RUN pip3 install --compile --no-cache-dir -r /app/requirements.txt

#replace with a volume
#COPY baseball.sql /app/mariadb/baseball.sql

COPY Assignment5.sql /app/Assignment5.sql

COPY load_baseball_data.sh /app/load_baseball_data.sh

#COPY Assignment5.py /app/Assignment5.py
COPY Final.py /app/Final.py


CMD ./load_baseball_data.sh

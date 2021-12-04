
FROM mariadb:latest

ENV PACKAGES openssh-server openssh-client
RUN apt-get update && apt-get install -y $PACKAGES

COPY baseball.sql .
COPY Assignment2.sql .

COPY load_baseball_data.sh .
RUN chmod u+x load_baseball_data.sh

RUN ./load_baseball_data.sh








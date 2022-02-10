#!/bin/bash
pwd

sleep 30

#Got some help from Will but wasnt able to figure it out.
if ! mysql -h mariadb -u root -pBDAMaster -e 'use baseball'; then
  echo "Loading baseball data into SQL"
  mysql -h mariadb -u root -pBDAMaster -e "create database baseball;"
  mysql -h mariadb -u root -pBDAMaster baseball < ./baseball.sql
fi

echo "Running Assignment5.sql"
mysql -h mariadb -u root -pBDAMaster baseball < ./Assignment5.sql

mkdir -p /app/html_files/plots

#echo "Running Assignment5.py"
#python ./Assignment5.py

echo "Running Final.py"
python ./Final.py

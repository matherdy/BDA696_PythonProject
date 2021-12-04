#!/bin/bash

pwd
docker-compose up mariadb


#Got some help from Will but wasnt able to figure it out.
if ! mysql -h mariadb -uroot -psecret -e 'use baseball'; then
  mysql -h mariadb -uroot -psecret -e "create database baseball;"
  mysql -h maraidb -uroot -psecret -D baseball < ./baseball.sql
fi
docker-compose up assignment5


mysql -h assignment5 -uroot -psecret baseball < ./Assignment5.sql

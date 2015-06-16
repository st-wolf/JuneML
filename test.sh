#!/bin/bash

###########################################################
# Прогоняет все тесты из указанной папки с помощью unittest
###########################################################

# Имя папки с тестами
TEST_FOLDER_NAME='tests'

for test_file in ./${TEST_FOLDER_NAME}/*.py;
do
	# echo ${TEST_FOLDER_NAME}/"${test_file##*/}"
	python3 -m unittest ${TEST_FOLDER_NAME}/"${test_file##*/}"
done
help:
	@echo 'Makefile for birbcam                                                      '
	@echo '                                                                          '
	@echo 'Usage:                                                                    '
	@echo '   make up                             Start birbcamm                     '
	@echo '   make build                          Build the docker containers        '
	@echo '   make create_db                      Create the modeldb                 '
	@echo '                                                                          '


up:
	docker-compose up

start:
	docker-compose up --detach

build:
	docker-compose build --parallel

create_db:
	docker-compose run webapp python3 util.py create_db


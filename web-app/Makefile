.POSIX:

DOCKER_WORKDIR = /usr/src/app
NAME_CURRENT_DIR = $(notdir $(shell pwd))

all: .venv main.py
	. $</bin/activate; python3 main.py

.venv: requirements.txt
	rm -rf $@/
	python3 -m venv $@/
	. $@/bin/activate; pip install -U pip wheel; pip install -Ur $<

clean:
	rm -rf __pycache__/ .venv/

docker:
	docker build -t $(NAME_CURRENT_DIR) .
	docker run --rm \
		--user $(shell id -u):$(shell id -g) \
		-w $(DOCKER_WORKDIR) \
		-v $(PWD):$(DOCKER_WORKDIR) \
		-p 7860:7860 \
		$(NAME_CURRENT_DIR) \
		python3 main.py

.POSIX:

ARGS= 
DEBUG_ARGS=--interactive --tty
MAKEFILE_DIR=$(dir $(realpath Makefile))
ifeq (, $(shell which nvidia-smi))
	DOCKER_GPU_ARGS=
else
	DOCKER_GPU_ARGS=--gpus all
endif

ms.pdf: ms.tex ms.bib results/completed
	docker container run \
		--rm \
		--user $(shell id -u):$(shell id -g) \
		--volume $(MAKEFILE_DIR):/home/latex \
		ghcr.io/pbizopoulos/texlive-full \
		latexmk -usepretex="\pdfinfoomitdate=1\pdfsuppressptexinfo=-1\pdftrailerid{}" -gg -pdf -cd /home/latex/ms.tex
	@if [ -f cache/.tmp.pdf ]; then \
		cmp ms.pdf cache/.tmp.pdf && echo 'ms.pdf unchanged.' || echo 'ms.pdf changed.'; fi
	@cp ms.pdf cache/.tmp.pdf

results/completed: Dockerfile requirements.txt $(shell find . -maxdepth 1 -name '*.py')
	rm -rf results/*
	docker image build --tag comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-in-ct .
	docker container run \
		$(DEBUG_ARGS) \
		--rm \
		--user $(shell id -u):$(shell id -g) \
		--volume $(MAKEFILE_DIR):/usr/src/app \
		$(DOCKER_GPU_ARGS) comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-in-ct \
		python main.py $(ARGS)
	touch results/completed

clean:
	rm -rf __pycache__/ cache/* results/* ms.bbl
	docker container run \
		--rm \
		--user $(shell id -u):$(shell id -g) \
		--volume $(MAKEFILE_DIR):/home/latex \
		ghcr.io/pbizopoulos/texlive-full \
		latexmk -C -cd /home/latex/ms.tex

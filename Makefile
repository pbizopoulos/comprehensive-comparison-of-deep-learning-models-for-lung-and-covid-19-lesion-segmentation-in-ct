.POSIX:

is_interactive:=$(shell [ -t 0 ] && echo 1)
ifdef is_interactive
	debug_args=--interactive --tty
endif

ifneq (, $(shell which nvidia-container-cli))
	gpu_args=--gpus all
endif

host_volume=$(dir $(realpath Makefile))
container_volume=/workspace

$(shell mkdir -p tmp/)

tmp/ms.pdf: ms.bib ms.tex tmp/execute-python
	docker container run \
		--rm \
		--user `id -u`:`id -g` \
		--volume $(host_volume):$(container_volume)/ \
		--workdir $(container_volume)/ \
		texlive/texlive latexmk -gg -pdf -usepretex="\pdfinfoomitdate=1\pdfsuppressptexinfo=-1\pdftrailerid{}" -outdir=tmp/ ms.tex

tmp/execute-python: Dockerfile main.py requirements.txt
	docker container run \
		$(debug_args) \
		$(gpu_args) \
		--detach-keys "ctrl-^,ctrl-^"  \
		--env HOME=$(container_volume)/tmp \
		--rm \
		--user `id -u`:`id -g` \
		--volume $(host_volume):$(container_volume)/ \
		--workdir $(container_volume)/ \
		`docker image build -q .` python main.py $(VERSION)
	touch tmp/execute-python

clean:
	rm -rf tmp/

tmp/format-python: main.py
	docker container run \
		--rm \
		--user `id -u`:`id -g` \
		--volume $(host_volume):$(container_volume)/ \
		--workdir $(container_volume)/ \
		alphachai/isort main.py
	docker container run \
		--rm \
		--user `id -u`:`id -g` \
		--volume $(host_volume):$(container_volume)/ \
		--workdir $(container_volume)/ \
		peterevans/autopep8 -i --max-line-length 1000 main.py
	touch tmp/format-python

tmp/lint-texlive: ms.bib ms.tex tmp/execute-python
	docker container run \
		--rm \
		--user `id -u`:`id -g` \
		--volume $(host_volume):$(container_volume)/ \
		--workdir $(container_volume)/ \
		texlive/texlive bash -c "chktex ms.tex && lacheck ms.tex"
	touch tmp/lint-texlive

tmp/arxiv.tar: tmp/ms.pdf
	cp tmp/ms.bbl .
	tar cf tmp/arxiv.tar ms.bbl ms.bib ms.tex `grep './tmp' tmp/ms.fls | uniq | cut -b 9-`
	rm ms.bbl

tmp/download-arxiv:
	curl https://arxiv.org/e-print/`grep arxiv.org README | cut -d '/' -f5` | tar xz
	mv ms.bbl tmp/
	touch tmp/download-arxiv tmp/execute-python

tmp/update-makefile:
	curl -LO https://github.com/pbizopoulos/a-makefile-for-developing-containerized-latex-technical-documents-template/raw/master/Makefile

tmp/update-texlive:
	docker image pull texlive/texlive

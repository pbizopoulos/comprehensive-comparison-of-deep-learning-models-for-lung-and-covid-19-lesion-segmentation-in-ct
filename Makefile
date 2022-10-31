.POSIX:

.PHONY: all check clean help

bib_file_name = ms.bib
container_engine = docker
tex_file_name = ms.tex
user_arg = $$(test $(container_engine) = 'docker' && printf '%s' "--user $$(id -u):$$(id -g)")
work_dir = /work

all: bin/all

check: bin/check

clean:
	rm -rf bin/

help:
	@printf 'make all	# Build binaries.\n'
	@printf 'make check	# Check code.\n'
	@printf 'make clean	# Remove binaries.\n'
	@printf 'make help	# Show help.\n'

$(bib_file_name):
	touch $(bib_file_name)

$(tex_file_name):
	printf "\\\documentclass{article}\n\n\\\begin{document}\nTitle\n\\\end{document}\n" > $(tex_file_name)

.dockerignore:
	printf '*\n' > .dockerignore

.gitignore:
	printf 'bin/\n' > .gitignore

bin:
	mkdir bin

bin/all: $(bib_file_name) $(tex_file_name) .dockerignore .gitignore bin
	$(container_engine) container run \
		$(user_arg) \
		--rm \
		--volume $$(pwd):$(work_dir)/ \
		--workdir $(work_dir)/ \
		texlive/texlive latexmk -gg -pdf -outdir=bin/ $(tex_file_name)
	touch bin/all

bin/check: $(tex_file_name) .dockerignore .gitignore bin
	$(container_engine) container run \
		$(user_arg) \
		--rm \
		--volume $$(pwd):$(work_dir)/ \
		--workdir $(work_dir)/ \
		texlive/texlive /bin/bash -c '\
		chktex $(tex_file_name) && \
		lacheck $(tex_file_name)'
	touch bin/check

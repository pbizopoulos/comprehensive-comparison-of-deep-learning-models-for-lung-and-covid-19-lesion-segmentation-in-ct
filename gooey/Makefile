.POSIX:

is_shell_interactive:=$(shell [ -t 0 ] && echo 1)
ifdef is_shell_interactive
	debug_args=--interactive --tty
endif

tmpdir=tmp
workdir=/app

$(shell mkdir -p $(tmpdir))

cli: Dockerfile requirements.txt main.py
	docker container run \
		$(debug_args) \
		--env HOME=$(workdir)/$(tmpdir) \
		--env TORCH_HOME=$(workdir)/$(tmpdir) \
		--rm \
		--user `id -u`:`id -g` \
		--volume `pwd`:$(workdir) \
		--workdir $(workdir) \
		`docker image build -q .` python3 main.py --ignore-gooey covid-19-pneumonia-4.jpg lung-segmentation Unet mobilenet_v2 imagenet

gui: Dockerfile requirements.txt main.py
	xhost +local:$(USER)
	docker container run \
		--env DISPLAY \
		--env HOME=$(workdir)/$(tmpdir) \
		--env MPLBACKEND=WXAgg \
		--env TORCH_HOME=$(workdir)/$(tmpdir) \
		--rm \
		--user `id -u`:`id -g` \
		--volume /tmp/.X11-unix:/tmp/.X11-unix:ro \
		--volume `pwd`:$(workdir) \
		--workdir $(workdir) \
		`docker image build -q .` python3 main.py
	xhost -local:$(USER)

clean:
	rm -rf $(tmpdir)/ __pycache__/

run-linux: $(tmpdir)/dist/main
	./$(tmpdir)/dist/main

run-windows: $(tmpdir)/dist/main.exe
	xhost +local:$(USER)
	docker container run -it \
		--env DISPLAY \
		--rm \
		--volume `pwd`:$(workdir) \
		--volume /tmp/.X11-unix:/tmp/.X11-unix:ro \
		cdrx/pyinstaller-windows "wine $(workdir)/$(tmpdir)/dist/main.exe"
	xhost -local:$(USER)

$(tmpdir)/dist/main: Dockerfile requirements.txt main.py
	docker container run \
		--rm \
		--user `id -u`:`id -g` \
		--volume `pwd`:$(workdir) \
		--workdir $(workdir) \
		`docker image build -q .` python3 -m PyInstaller --runtime-tmpdir /var/tmp --hidden-import=torch --specpath $(tmpdir)/spec --clean --distpath $(tmpdir)/dist --workpath $(tmpdir)/build --onefile main.py

$(tmpdir)/dist/main.exe: requirements.txt main.py
	docker container run \
		--rm \
		--volume `pwd`:/src/ \
		cdrx/pyinstaller-windows "pyinstaller --hidden-import=torch --specpath $(tmpdir)/spec --clean --distpath $(tmpdir)/dist --workpath $(tmpdir)/build --onefile main.py"

$(tmpdir)/format-python: main.py
	docker container run \
		--rm \
		--user `id -u`:`id -g` \
		--volume `pwd`:$(workdir)/ \
		--workdir $(workdir)/ \
		alphachai/isort main.py
	docker container run \
		--rm \
		--user `id -u`:`id -g` \
		--volume `pwd`:$(workdir)/ \
		--workdir $(workdir)/ \
		peterevans/autopep8 -i --max-line-length 1000 main.py
	touch $(tmpdir)/format-python

$(tmpdir)/update-docker-images:
	docker image pull alphachai/isort
	docker image pull peterevans/autopep8

.POSIX:

tmpdir=tmp
workdir=/app

all:
	docker container run \
		--env HOME=$(workdir)/$(tmpdir) \
		--env TORCH_HOME=$(workdir)/$(tmpdir) \
		--publish 7860:7860 \
		--rm \
		--user `id -u`:`id -g` \
		--volume `pwd`:$(workdir)/ \
		--workdir $(workdir)/ \
		-it \
		`docker image build -q .` python main.py

clean:
	rm -rf $(tmpdir)

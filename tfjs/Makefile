.POSIX:

tmpdir=tmp
workdir=/app

$(tmpdir)/$(MODEL_NAME)/$(MODEL_NAME)/model.json: Dockerfile main.py requirements.txt
	mkdir -p $(tmpdir)/
	docker build -t test-image .
	docker container run \
		--detach-keys "ctrl-^,ctrl-^" \
		--env HOME=$(workdir)/$(tmpdir)/ \
		--rm \
		--user `id -u`:`id -g` \
		--volume `pwd`:$(workdir)/ \
		--workdir $(workdir)/ \
		`docker image build -q .` bash -c 'python main.py $(MODEL_NAME) && tensorflowjs_converter --input_format=tf_saved_model $(tmpdir)/$(MODEL_NAME)/model $(tmpdir)/$(MODEL_NAME)/$(MODEL_NAME)'

clean:
	rm -rf $(tmpdir)/

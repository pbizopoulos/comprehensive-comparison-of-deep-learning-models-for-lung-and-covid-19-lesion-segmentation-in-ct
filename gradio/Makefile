.POSIX:

container_volume=/workspace
tmp_directory=tmp

docker:
	docker container run \
		--rm \
		--env HOME=$(container_volume)/$(tmp_directory) \
		--env TORCH_HOME=$(container_volume)/$(tmp_directory) \
		--publish 7860:7860 \
		--user `id -u`:`id -g` \
		--volume `pwd`:$(container_volume)/ \
		--workdir $(container_volume)/ \
		`docker image build -q .` python main.py

clean:
	rm -rf tmp/

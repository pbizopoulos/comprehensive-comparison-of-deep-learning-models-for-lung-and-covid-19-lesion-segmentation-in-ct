.POSIX:

docker:
	docker container run --rm \
		--env HOME=/workspace/tmp \
		--env TORCH_HOME=/workspace/tmp \
		--publish 7860:7860 \
		--user `id -u`:`id -g` \
		--volume $(dir $(realpath Makefile)):/workspace/ \
		--workdir /workspace/ \
		`docker image build -q .` \
		python main.py

clean:
	rm -rf tmp/

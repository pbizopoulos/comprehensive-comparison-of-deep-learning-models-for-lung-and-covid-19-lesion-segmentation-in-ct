.POSIX:

docker:
	docker run --rm \
		--user `id -u`:`id -g` \
		--env HOME=/workspace/tmp \
		--env TORCH_HOME=/workspace/tmp \
		--workdir /workspace/ \
		--volume $(dir $(realpath Makefile)):/workspace/ \
		-p 7860:7860 \
		`docker image build -q .` \
		python main.py

clean:
	rm -rf tmp/

.POSIX:

DOCKER_WORKDIR = /usr/src/app
MAKEFILE_DIR=$(dir $(realpath Makefile))

docker:
	docker build -t comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-in-ct-demos .
	docker run --rm \
		--user $(shell id -u):$(shell id -g) \
		--env HOME=$(DOCKER_WORKDIR)/.cache \
		--env TORCH_HOME=$(DOCKER_WORKDIR)/.cache \
		--workdir $(DOCKER_WORKDIR) \
		--volume $(MAKEFILE_DIR):$(DOCKER_WORKDIR) \
		-p 7860:7860 \
		comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-in-ct-demos

clean:
	rm -rf __pycache__/

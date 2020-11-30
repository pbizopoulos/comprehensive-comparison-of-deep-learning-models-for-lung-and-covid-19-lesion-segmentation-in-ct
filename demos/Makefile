.POSIX:

docker:
	docker build -t comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-in-ct-demos .
	docker run --rm \
		--user `id -u`:`id -g` \
		--env HOME=/usr/src/app/tmp \
		--env TORCH_HOME=/usr/src/app/tmp \
		--workdir /usr/src/app \
		--volume $(dir $(realpath Makefile)):/usr/src/app \
		-p 7860:7860 \
		comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-in-ct-demos

clean:
	rm -rf tmp

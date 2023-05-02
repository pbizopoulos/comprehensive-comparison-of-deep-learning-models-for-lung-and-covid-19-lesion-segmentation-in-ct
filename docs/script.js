"use strict";
const imageFileReader = new FileReader();
const imageInputCanvas = document.getElementById("image-input-canvas");
const imageInputContext = imageInputCanvas.getContext("2d");
const maskOutputCanvas = document.getElementById("mask-output-canvas");
const maskOutputContext = maskOutputCanvas.getContext("2d");
const modelDownloadDiv = document.getElementById("model-download-div");
const modelDownloadProgress = document.getElementById("model-download-progress");
const modelSelect = document.getElementById("model-select");
const image = new Image();
let model;
image.crossOrigin = "anonymous";
image.onload = imageOnLoad;
image.src = "https://raw.githubusercontent.com/pbizopoulos/comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-in-ct/main/latex/python/assets/lung-segmentation-example-data.png";
imageFileReader.onload = imageFileReaderOnLoad;
inputFile.onchange = inputFileOnChange;

function disableUI(argument) {
	const nodes = document.getElementById("input-control-div").getElementsByTagName("*");
	for (let i = 0; i < nodes.length; i++) {
		nodes[i].disabled = argument;
	}
}

function inputFileOnChange(event) {
	const files = event.currentTarget.files;
	if (files[0]) {
		imageFileReader.readAsDataURL(files[0]);
	}
}

function imageFileReaderOnLoad() {
	image.src = imageFileReader.result;
}

function imageOnLoad() {
	imageInputContext.clearRect(0, 0, imageInputCanvas.width, imageInputCanvas.height);
	imageInputContext.drawImage(image, 0, 0, image.width, image.height, 0, 0, imageInputCanvas.width, imageInputCanvas.height);
	predictView();
}

async function loadModel(predictFunction) {
	modelDownloadDiv.style.display = "";
	const loadModelFunction = tf.loadGraphModel;
	model = await loadModelFunction(modelSelect.value, {
		onProgress: (fraction) => {
			modelDownloadProgress.value = fraction;
			if (fraction === 1) {
				modelDownloadDiv.style.display = "none";
			}
			disableUI(true);
		},
	});
	predictFunction();
	disableUI(false);
}

function predictView() {
	if (model === undefined) {
		return;
	}
	tf.tidy(() => {
		let fromPixels = tf.browser.fromPixels(imageInputCanvas);
		const originalShape = fromPixels.shape.slice(0, 2);
		fromPixels = tf.image.resizeNearestNeighbor(fromPixels, [model.inputs[0].shape[2], model.inputs[0].shape[3]]);
		let pixels = fromPixels.slice([0, 0, 2]).squeeze(-1).expandDims(0).expandDims(0);
		pixels = pixels.mul(3 / 255);
		pixels = pixels.sub(1.5);
		const mask = model.predict(pixels);
		let maskToPixels = mask.squeeze(0).squeeze(0);
		const alphaTensor = tf.tensor([0.3]);
		const alphaChannel = alphaTensor.where(maskToPixels.greaterEqual(0.5), 0);
		maskToPixels = tf.stack([maskToPixels, tf.zerosLike(maskToPixels), tf.zerosLike(maskToPixels), alphaChannel], -1);
		maskToPixels = tf.image.resizeNearestNeighbor(maskToPixels, originalShape);
		maskOutputContext.clearRect(0, 0, imageInputCanvas.width, imageInputCanvas.height);
		tf.browser.toPixels(maskToPixels, maskOutputCanvas);
	});
}

loadModel(predictView);

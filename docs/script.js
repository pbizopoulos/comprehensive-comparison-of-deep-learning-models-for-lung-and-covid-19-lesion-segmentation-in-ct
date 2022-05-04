'use strict';

const imageFileReader = new FileReader();
const imageInputCanvas = document.getElementById('imageInputCanvas');
const imageInputContext = imageInputCanvas.getContext('2d');
const imageOutputCanvas = document.getElementById('imageOutputCanvas');
const imageOutputContext = imageOutputCanvas.getContext('2d');
const maskOutputCanvas = document.getElementById('maskOutputCanvas');
const maskOutputContext = maskOutputCanvas.getContext('2d');
let image = new Image();
let model;

image.crossOrigin = 'anonymous';
image.src = 'https://raw.githubusercontent.com/pbizopoulos/comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-in-ct/master/docs/lung-segmentation-example-data.png';


function disableUI(argument) {
	const nodes = document.getElementById('inputControlDiv').getElementsByTagName('*');
	for(let i = 0; i < nodes.length; i++){
		nodes[i].disabled = argument;
	}
}

async function loadModel(predictFunction) {
	const loadModelFunction = tf.loadGraphModel;
	model = await loadModelFunction(modelSelect.value, {
		onProgress: function (fraction) {
			document.getElementById('modelDownloadFractionDiv').textContent = `Downloading model, please wait ${Math.round(100*fraction)}%.`;
			if (fraction == 1) {
				document.getElementById('modelDownloadFractionDiv').textContent = 'Model downloaded.';
			}
			disableUI(true);
		}
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
		pixels = pixels.mul(3/255);
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

image.onload = function() {
	imageInputContext.clearRect(0, 0, imageInputCanvas.width, imageInputCanvas.height);
	imageInputContext.drawImage(image, 0, 0, image.width, image.height, 0, 0, imageInputCanvas.width, imageInputCanvas.height);
	imageOutputContext.clearRect(0, 0, imageInputCanvas.width, imageInputCanvas.height);
	imageOutputContext.drawImage(image, 0, 0, image.width, image.height, 0, 0, imageInputCanvas.width, imageInputCanvas.height);
	predictView();
};

imageFileReader.onload = function() {
	image.src = imageFileReader.result;
};

imageInputCanvas.onmousedown = function(event) {
	imageInputContext.clearRect(0, 0, this.width, this.height);
	imageInputContext.save();
	imageOutputContext.clearRect(0, 0, this.width, this.height);
	imageOutputContext.save();
	const rectangular = imageInputCanvas.getBoundingClientRect();
	const mousedownX = event.clientX - rectangular.left;
	const mousedownY = event.clientY - rectangular.top;
	let mousedownDistanceFromCenter = Math.sqrt((mousedownX - this.width/2)**2 + (mousedownY - this.height/2)**2);
	let rotationDegree = 0;
	if (mousedownDistanceFromCenter > this.width/4) {
		const mousedownXtranslated = (mousedownX - this.width/2);
		const mousedownYtranslated = (mousedownY - this.height/2);
		const originXtranslated = (this.width/2 - this.width/2);
		const originYtranslated = (0 - this.height/2);
		const dot = mousedownXtranslated*originXtranslated + mousedownYtranslated*originYtranslated;
		const det = mousedownYtranslated*originXtranslated - mousedownXtranslated*originYtranslated;
		rotationDegree = Math.atan2(det, dot);
	}
	imageInputContext.translate(this.width/2, this.height/2);
	imageInputContext.rotate(rotationDegree);
	imageInputContext.drawImage(image, 0, 0, image.width, image.height, -this.width/2, -this.height/2, this.width, this.height);
	imageInputContext.restore();
	imageOutputContext.translate(this.width/2, this.height/2);
	imageOutputContext.rotate(rotationDegree);
	imageOutputContext.drawImage(image, 0, 0, image.width, image.height, -this.width/2, -this.height/2, this.width, this.height);
	imageOutputContext.restore();
	predictView();
}

inputFile.onchange = function(event) {
	const files = event.currentTarget.files;
	if (files[0]) {
		imageFileReader.readAsDataURL(files[0]);
	}
}

loadModel(predictView);

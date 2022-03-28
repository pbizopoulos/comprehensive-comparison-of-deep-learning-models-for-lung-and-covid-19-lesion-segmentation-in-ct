const canvasWidth = 256;
const canvasHeight = 256;
const inputFilename = "https://raw.githubusercontent.com/pbizopoulos/comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-in-ct/master/docs/lung-segmentation-example-data.png";
const canvasImageInput = document.getElementById("canvasImageInput");
const contextImageInput = canvasImageInput.getContext("2d");
const canvasImageOutput = document.getElementById("canvasImageOutput");
const contextImageOutput = canvasImageOutput.getContext("2d");
const canvasMaskOutput = document.getElementById("canvasMaskOutput");
const contextMaskOutput = canvasMaskOutput.getContext("2d");

function predictView() {
	if (model === undefined) {
		return;
	}
	tf.tidy(() => {
		let fromPixels = tf.browser.fromPixels(canvasImageInput);
		originalShape = fromPixels.shape.slice(0, 2);
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
		contextMaskOutput.clearRect(0, 0, canvasWidth, canvasHeight);
		tf.browser.toPixels(maskToPixels, canvasMaskOutput);
	});
}


function imageLoadView() {
	const files = event.currentTarget.files;
	if (files[0]) {
		imageFileReader.readAsDataURL(files[0]);
	}
}

canvasImageInput.onmousedown = function(event) {
	contextImageInput.clearRect(0, 0, canvasWidth, canvasHeight);
	contextImageInput.save();
	contextImageOutput.clearRect(0, 0, canvasWidth, canvasHeight);
	contextImageOutput.save();
	const rect = canvasImageInput.getBoundingClientRect();
	const mousedownX = event.clientX - rect.left;
	const mousedownY = event.clientY - rect.top;
	let mousedownDistanceFromCenter = Math.sqrt((mousedownX - canvasWidth/2)**2 + (mousedownY - canvasHeight/2)**2);
	let rotationDegree = 0;
	if (mousedownDistanceFromCenter > canvasWidth/4) {
		const mousedownXtranslated = (mousedownX - canvasWidth/2);
		const mousedownYtranslated = (mousedownY - canvasHeight/2);
		const originXtranslated = (canvasWidth/2 - canvasWidth/2);
		const originYtranslated = (0 - canvasHeight/2);
		const dot = mousedownXtranslated*originXtranslated + mousedownYtranslated*originYtranslated;
		const det = mousedownYtranslated*originXtranslated - mousedownXtranslated*originYtranslated;
		rotationDegree = Math.atan2(det, dot);
	}
	contextImageInput.translate(canvasWidth/2, canvasHeight/2);
	contextImageInput.rotate(rotationDegree);
	contextImageInput.drawImage(imageInput, 0, 0, imageInput.width, imageInput.height, -canvasWidth/2, -canvasHeight/2, canvasWidth, canvasHeight);
	contextImageInput.restore();
	contextImageOutput.translate(canvasWidth/2, canvasHeight/2);
	contextImageOutput.rotate(rotationDegree);
	contextImageOutput.drawImage(imageInput, 0, 0, imageInput.width, imageInput.height, -canvasWidth/2, -canvasHeight/2, canvasWidth, canvasHeight);
	contextImageOutput.restore();
	predictView();
}

let imageInput = new Image();
const imageFileReader = new FileReader();
imageFileReader.onload = () => {
	imageInput.src = imageFileReader.result;
};

imageInput.crossOrigin = 'anonymous';
imageInput.src = inputFilename;
imageInput.onload = () => {
	contextImageInput.clearRect(0, 0, canvasWidth, canvasHeight);
	contextImageInput.drawImage(imageInput, 0, 0, imageInput.width, imageInput.height, 0, 0, canvasWidth, canvasHeight);
	contextImageOutput.clearRect(0, 0, canvasWidth, canvasHeight);
	contextImageOutput.drawImage(imageInput, 0, 0, imageInput.width, imageInput.height, 0, 0, canvasWidth, canvasHeight);
	predictView();
};


function disableUI(argument) {
	const nodes = document.getElementById('divInputControl').getElementsByTagName('*');
	for(let i = 0; i < nodes.length; i++){
		nodes[i].disabled = argument;
	}
}

let model;
async function loadModel(predictFunction) {
	const loadModelFunction = tf.loadGraphModel;
	model = await loadModelFunction(selectModel.value, {
		onProgress: function (fraction) {
			document.getElementById('divModelDownloadFraction').innerHTML = 'Downloading model, please wait ' + Math.round(100*fraction) + '%.';
			if (fraction == 1) {
				document.getElementById('divModelDownloadFraction').innerHTML = 'Model downloaded.';
			}
			disableUI(true);
		}
	});
	predictFunction();
	disableUI(false);
}
loadModel(predictView);


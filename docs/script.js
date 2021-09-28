'use strict';

function loadImages() {
	disableUI(true);
	files = event.currentTarget.files;
	if (files[0] == undefined) {
		disableUI(false);
		return;
	}
	resetData();
	if (files[0].name.includes('.nii')) {
		fileName = files[0].name.split('.nii')[0];
		readNiiFile(files[0]);
	} else {
		fileName = files[0].name;
		itk.readImageDICOMFileSeries(files)
			.then(function ({ image }) {
				itk.writeImageArrayBuffer(null, false, image, 'unnamed.nii')
					.then((data) => {
						const blob = new Blob([data.arrayBuffer]);
						readNiiFile(blob);
					});
			});
	}
}

function selectModelName() {
	const [experiment, architecture, encoder, encoderWeights] = selectModel.value.split('.');
	(async function () {
		selectedModel = await tf.loadGraphModel(`https://raw.githubusercontent.com/pbizopoulos/lung-and-covid-19-lesion-segmentation-models-tfjs/master/${experiment}.${architecture}.${encoder}.${encoderWeights}/model.json`, {onProgress: disableUI(true)});
		disableUI(false);
	})();
}

function predictMask() {
	let imageSlice = images.slice(imageOffset, imageOffset + imageSize);
	imageSlice = new Float32Array(imageSlice);
	tf.tidy(() => {
		let imageSliceTensor = tf.tensor(imageSlice);
		imageSliceTensor = tf.reshape(imageSliceTensor, [rows, columns]);
		if (imageSliceTensor.shape[0] !== 512 || imageSliceTensor.shape[1] !== 512) {
			imageSliceTensor = imageSliceTensor.expandDims(-1);
			imageSliceTensor = tf.image.resizeBilinear(imageSliceTensor, [512, 512]);
			imageSliceTensor = imageSliceTensor.squeeze(-1);
		}
		imageSliceTensor = tf.add(imageSliceTensor, 500);
		imageSliceTensor = tf.div(imageSliceTensor, 500);
		const expandedTensor = imageSliceTensor.expandDims(0).expandDims(0);
		let modelPrediction = selectedModel.predict(expandedTensor);
		if (modelPrediction.size !== imageSize) {
			modelPrediction = modelPrediction.reshape([512, 512, 1]);
			modelPrediction = tf.image.resizeNearestNeighbor(modelPrediction, [rows, columns]);
		}
		modelPrediction = modelPrediction.round().dataSync();
		for (let i = 0; i < modelPrediction.length; i++) {
			masks[imageOffset + i] = modelPrediction[i];
		}
	});
	visualizeImageDataMask();
}

function saveMasks() {
	if (files == undefined) {
		return;
	}
	const niftiHeaderTmp = decompressedFile.slice(0, 352);
	const data = [new Uint8Array(niftiHeaderTmp, 0, niftiHeaderTmp.length), new Uint8Array(masks.buffer, 0, masks.buffer.length)];
	saveData(data, `${fileName}-masks.nii`);
}

function readNiiFile(file) {
	const reader = new FileReader();
	reader.onloadend = function (event) {
		if (event.target.readyState === FileReader.DONE) {
			let niftiHeader;
			let niftiImage;
			if (nifti.isCompressed(event.target.result)) {
				decompressedFile = nifti.decompress(event.target.result);
			} else {
				decompressedFile = event.target.result;
			}
			if (nifti.isNIFTI(decompressedFile)) {
				niftiHeader = nifti.readHeader(decompressedFile);
				niftiImage = nifti.readImage(niftiHeader, decompressedFile);
			}
			switch (niftiHeader.datatypeCode) {
				case nifti.NIFTI1.TYPE_UINT8:
					images = new Uint8Array(niftiImage);
					masks = new Uint8Array(images.length);
					break;
				case nifti.NIFTI1.TYPE_INT16:
					images = new Int16Array(niftiImage);
					masks = new Int16Array(images.length);
					break;
				case nifti.NIFTI1.TYPE_INT32:
					images = new Int32Array(niftiImage);
					masks = new Int32Array(images.length);
					break;
				case nifti.NIFTI1.TYPE_FLOAT32:
					images = new Float32Array(niftiImage);
					masks = new Float32Array(images.length);
					break;
				case nifti.NIFTI1.TYPE_FLOAT64:
					images = new Float64Array(niftiImage);
					masks = new Float64Array(images.length);
					break;
				case nifti.NIFTI1.TYPE_INT8:
					images = new Int8Array(niftiImage);
					masks = new Int8Array(images.length);
					break;
				case nifti.NIFTI1.TYPE_UINT16:
					images = new Uint16Array(niftiImage);
					masks = new Uint16Array(images.length);
					break;
				case nifti.NIFTI1.TYPE_UINT32:
					images = new Uint32Array(niftiImage);
					masks = new Uint32Array(images.length);
					break;
				default:
					return;
			}
			numImages = niftiHeader.dims[3] - 1;
			rows = niftiHeader.dims[2];
			columns = niftiHeader.dims[1];
			imageSize = rows * columns;
			const imageSlice = images.slice(0, imageSize);
			let max = -Infinity;
			let min = Infinity;
			for (let i = 0; i < imageSlice.length; i++) {
				if (imageSlice[i] > max) {
					max = imageSlice[i];
				}
				if (imageSlice[i] < min) {
					min = imageSlice[i];
				}
			}
			imageValueMin = min;
			imageValueRange = (max - min) / 255;
			updateUI();
			visualizeImageDataImage();
		}
		disableUI(false);
	};
	reader.readAsArrayBuffer(file);
}

function visualizeImageDataImage() {
	const imageDataImage = new ImageData(columns, rows);
	const imageDataImageData = imageDataImage.data;
	for (let i = 0; i < imageSize; i++) {
		const imageValue = (images[imageOffset + i] - imageValueMin) / imageValueRange;
		imageDataImageData[4*i] = imageValue;
		imageDataImageData[4*i + 1] = imageValue;
		imageDataImageData[4*i + 2] = imageValue;
		imageDataImageData[4*i + 3] = 255;
	}
	contextImage.putImageData(imageDataImage, 0, 0);
}

function visualizeImageDataMask() {
	const imageDataMask = new ImageData(columns, rows);
	const imageDataMaskData = imageDataMask.data;
	for (let i = 0; i < imageSize; i++) {
		if (masks[imageOffset + i] === 1) {
			imageDataMaskData[4*i] = 255;
			imageDataMaskData[4*i + 1] = 0;
			imageDataMaskData[4*i + 2] = 0;
			imageDataMaskData[4*i + 3] = 100;
		}
	}
	contextMask.putImageData(imageDataMask, 0, 0);
}

function disableUI(argument) {
	document.getElementById('buttonPredictMask').disabled = argument;
	document.getElementById('buttonSaveMasks').disabled = argument;
	document.getElementById('inputFile').disabled = argument;
	selectModel.disabled = argument;
}

function updateUI() {
	document.getElementById('divImageIndex').innerHTML = `Image index: ${imageIndex}/${numImages}`;
	document.getElementById('divRowsXColumns').innerHTML = `RowsXColumns: ${rows}x${columns}`;
}

function saveData(data, fileName) {
	const a = document.createElement('a');
	document.body.appendChild(a);
	a.style = 'display: none';
	const blob = new Blob(data);
	const url = window.URL.createObjectURL(blob);
	a.href = url;
	a.download = fileName;
	a.click();
	window.URL.revokeObjectURL(url);
}

function resetData() {
	columns = 0;
	decompressedFile = null;
	imageIndex = 0;
	imageOffset = 0;
	imageSize = columns * rows;
	imageValueMin = 0;
	imageValueRange = 1;
	images = new Uint8Array(imageSize);
	masks = new Uint8Array(imageSize);
	numImages = 0;
	rows = 0;
}

window.addEventListener('keydown', function (event) {
	if (event.key === 'ArrowDown' && (imageIndex > 0)) {
		imageIndex--;
	} else if (event.key === 'ArrowUp' && (imageIndex < numImages)) {
		imageIndex++;
	} else {
		return;
	}
	imageOffset = imageSize * imageIndex;
	updateUI();
	visualizeImageDataImage();
	visualizeImageDataMask();
});

const canvasImage = document.getElementById('canvasImage');
const canvasMask = document.getElementById('canvasMask');
const contextImage = canvasImage.getContext('2d');
const contextMask = canvasMask.getContext('2d');
const selectModel = document.getElementById('selectModel');

const modelNames = [
	'lesion-segmentation-a.FPN.mobilenet_v2.imagenet',
	'lesion-segmentation-a.FPN.resnet18.imagenet',
	'lesion-segmentation-a.FPN.vgg11.imagenet',
	'lesion-segmentation-a.FPN.vgg13.imagenet',
	'lesion-segmentation-a.Linknet.mobilenet_v2.imagenet',
	'lesion-segmentation-a.Linknet.resnet18.imagenet',
	'lesion-segmentation-a.Linknet.vgg11.imagenet',
	'lesion-segmentation-a.Linknet.vgg13.imagenet',
	'lesion-segmentation-a.Unet.mobilenet_v2.imagenet',
	'lung-segmentation.FPN.mobilenet_v2.imagenet',
	'lung-segmentation.FPN.resnet18.imagenet',
	'lung-segmentation.FPN.vgg11.imagenet',
	'lung-segmentation.FPN.vgg13.imagenet',
	'lung-segmentation.Linknet.mobilenet_v2.imagenet',
	'lung-segmentation.Linknet.resnet18.imagenet',
	'lung-segmentation.Linknet.vgg11.imagenet',
	'lung-segmentation.Linknet.vgg13.imagenet',
	'lung-segmentation.Unet.mobilenet_v2.imagenet'
];

for (const modelName of modelNames)
{
	const option = document.createElement('option');
	option.text = modelName;
	selectModel.appendChild(option);
}

let columns;
let decompressedFile;
let fileName;
let files;
let imageIndex;
let imageOffset;
let imageSize;
let imageValueMin;
let imageValueRange;
let images;
let masks;
let numImages;
let rows;
let selectedModel;

resetData();
updateUI();
selectModelName();

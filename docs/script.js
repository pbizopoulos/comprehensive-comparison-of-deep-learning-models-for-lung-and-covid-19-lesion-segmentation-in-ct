const helpButton = document.getElementById("help-button");
const helpDialog = document.getElementById("help-dialog");
const imageInputCanvas = document.getElementById("image-input-canvas");
const imageInputContext = imageInputCanvas.getContext("2d");
const inputFile = document.getElementById("input-file");
const loadingDialog = document.getElementById("loading-dialog");
const maskOutputCanvas = document.getElementById("mask-output-canvas");
const maskOutputContext = maskOutputCanvas.getContext("2d");
const modelSelect = document.getElementById("model-select");
let session;
const image = new Image();
image.crossOrigin = "anonymous";
image.onload = imageOnLoad;
image.src =
	"https://raw.githubusercontent.com/pbizopoulos/comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-in-ct/main/docs/prm/lung-segmentation-example-data.png";
const imageFileReader = new FileReader();
imageFileReader.onload = imageFileReaderOnLoad;
inputFile.onchange = inputFileOnChange;

helpButton.addEventListener("click", () => {
	helpDialog.showModal();
});

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
	imageInputContext.clearRect(
		0,
		0,
		imageInputCanvas.width,
		imageInputCanvas.height,
	);
	imageInputContext.drawImage(
		image,
		0,
		0,
		image.width,
		image.height,
		0,
		0,
		imageInputCanvas.width,
		imageInputCanvas.height,
	);
	predictView();
}

async function loadModel(predictFunction) {
	loadingDialog.showModal();
	session = await ort.InferenceSession.create(modelSelect.value);
	loadingDialog.close();
	predictFunction();
}

async function predictView() {
	if (session === undefined) {
		return;
	}
	let fromPixels = tf.browser.fromPixels(imageInputCanvas);
	const originalShape = fromPixels.shape.slice(0, 2);
	fromPixels = tf.image.resizeNearestNeighbor(fromPixels, [512, 512]);
	let pixels = fromPixels
		.slice([0, 0, 2])
		.squeeze(-1)
		.expandDims(0)
		.expandDims(0);
	pixels = pixels.div(255);
	const tensorA = new ort.Tensor(
		"float32",
		pixels.dataSync(),
		[1, 1, 512, 512],
	);
	const feeds = { "x.1": tensorA };
	const results = await session.run(feeds);
	const mask = results["729"].cpuData;
	let maskToPixels = tf.tensor(mask, [512, 512]);
	const alphaTensor = tf.tensor([0.3]);
	const alphaChannel = alphaTensor.where(maskToPixels.greaterEqual(0.5), 0);
	maskToPixels = tf.stack(
		[
			maskToPixels,
			tf.zerosLike(maskToPixels),
			tf.zerosLike(maskToPixels),
			alphaChannel,
		],
		-1,
	);
	maskToPixels = tf.image.resizeNearestNeighbor(maskToPixels, originalShape);
	maskOutputContext.clearRect(
		0,
		0,
		imageInputCanvas.width,
		imageInputCanvas.height,
	);
	tf.browser.toPixels(maskToPixels.clipByValue(0, 1), maskOutputCanvas);
}

loadModel(predictView);

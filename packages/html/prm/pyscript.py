"""Comprehensive Comparison of Deep Learning Models for Lung and COVID-19 Lesion Segmentation in CT Gradio."""  # noqa: E501,INP001

import cv2
import gradio as gr
import numpy as np
import numpy.typing as npt


def _submit(img: npt.NDArray[np.float64], model_name: str) -> npt.NDArray[np.float64]:
    if model_name == "lesion segmentation":
        model_file_name = "model-lesion-segmentation-a-FPN-mobilenet_v2-imagenet.onnx"
    elif model_name == "lung segmentation":
        model_file_name = "model-lung-segmentation-FPN-mobilenet_v2-imagenet.onnx"
    img_cv2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_cv2 = cv2.normalize(img_cv2, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    net = cv2.dnn.readNet(model_file_name)
    blob = cv2.dnn.blobFromImage(img_cv2, 1.0, (512, 512), (0, 0, 0))
    net.setInput(blob)
    mask = net.forward()
    return cv2.resize(mask.squeeze(), img.shape[:2])


gr.Interface(
    _submit,
    [
        gr.Image(),
        gr.Dropdown(
            ["lesion segmentation", "lung segmentation"],
            label="Model",
            value="lesion segmentation",
        ),
    ],
    gr.Image(),
    flagging_mode="never",
    analytics_enabled=False,
    examples=[
        ["lesion-segmentation-example-data.jpg", "lesion segmentation"],
        ["lung-segmentation-example-data.png", "lung segmentation"],
    ],
).launch()

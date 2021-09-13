import argparse
import os

from onnx_tf.backend import prepare
import onnx
import tensorflow as tf
import torch


if __name__ == '__main__':
    torch.hub.set_dir('tmp')
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name')
    args = parser.parse_args()
    [experiment, architecture, encoder, encoder_weights] = args.model_name.split('.')
    filePath = f'tmp/{args.model_name}'
    os.makedirs(filePath, exist_ok=True)
    model = torch.hub.load('pbizopoulos/comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-in-ct', 'segmentation_model', experiment, architecture, encoder, encoder_weights)
    example_input = torch.randn(1, 1, 512, 512, requires_grad=False)
    torch.onnx.export(model, example_input, f'{filePath}/model.onnx', export_params=True, opset_version=11)
    onnx_model = onnx.load(f'{filePath}/model.onnx')
    tf_model = prepare(onnx_model)
    tf_model.export_graph(f'{filePath}/model')

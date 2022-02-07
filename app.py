from pyclientsideml import generate_page


def main():
    ml_type = 'image-segmentation'
    model_dirs = [
            'https://raw.githubusercontent.com/pbizopoulos/comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-inct-tfjs/master/lesion-segmentation-a.FPN.mobilenet_v2.imagenet/model.json',
            'https://raw.githubusercontent.com/pbizopoulos/comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-inct-tfjs/master/lesion-segmentation-a.FPN.resnet18.imagenet/model.json',
            'https://raw.githubusercontent.com/pbizopoulos/comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-inct-tfjs/master/lesion-segmentation-a.FPN.vgg11.imagenet/model.json',
            'https://raw.githubusercontent.com/pbizopoulos/comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-inct-tfjs/master/lesion-segmentation-a.FPN.vgg13.imagenet/model.json',
            'https://raw.githubusercontent.com/pbizopoulos/comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-inct-tfjs/master/lesion-segmentation-a.Linknet.mobilenet_v2.imagenet/model.json',
            'https://raw.githubusercontent.com/pbizopoulos/comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-inct-tfjs/master/lesion-segmentation-a.Linknet.resnet18.imagenet/model.json',
            'https://raw.githubusercontent.com/pbizopoulos/comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-inct-tfjs/master/lesion-segmentation-a.Linknet.vgg11.imagenet/model.json',
            'https://raw.githubusercontent.com/pbizopoulos/comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-inct-tfjs/master/lesion-segmentation-a.Linknet.vgg13.imagenet/model.json',
            'https://raw.githubusercontent.com/pbizopoulos/comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-inct-tfjs/master/lesion-segmentation-a.Unet.mobilenet_v2.imagenet/model.json',
            'https://raw.githubusercontent.com/pbizopoulos/comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-inct-tfjs/master/lung-segmentation.FPN.mobilenet_v2.imagenet/model.json',
            'https://raw.githubusercontent.com/pbizopoulos/comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-inct-tfjs/master/lung-segmentation.FPN.resnet18.imagenet/model.json',
            'https://raw.githubusercontent.com/pbizopoulos/comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-inct-tfjs/master/lung-segmentation.FPN.vgg11.imagenet/model.json',
            'https://raw.githubusercontent.com/pbizopoulos/comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-inct-tfjs/master/lung-segmentation.FPN.vgg13.imagenet/model.json',
            'https://raw.githubusercontent.com/pbizopoulos/comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-inct-tfjs/master/lung-segmentation.Linknet.mobilenet_v2.imagenet/model.json',
            'https://raw.githubusercontent.com/pbizopoulos/comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-inct-tfjs/master/lung-segmentation.Linknet.resnet18.imagenet/model.json',
            'https://raw.githubusercontent.com/pbizopoulos/comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-inct-tfjs/master/lung-segmentation.Linknet.vgg11.imagenet/model.json',
            'https://raw.githubusercontent.com/pbizopoulos/comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-inct-tfjs/master/lung-segmentation.Linknet.vgg13.imagenet/model.json',
            'https://raw.githubusercontent.com/pbizopoulos/comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-inct-tfjs/master/lung-segmentation.Unet.mobilenet_v2.imagenet/model.json']
    output_filename = 'index.html'
    class_names = ['covid-19']
    input_filename = 'https://raw.githubusercontent.com/pbizopoulos/comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-in-ct/master/example-image-1.jpg'
    title = 'COVID-19 CT segmentation demo'
    description = 'NOT FOR MEDICAL USE. Choose a lung CT image (.jpg,.png,.gif) and segment COVID-19 lesions using a DNN.'
    url = 'https://github.com/pbizopoulos/comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-in-ct'
    block_width = 256
    block_height = 256
    pixel_scaling = 3/255
    pixel_baseline = 1.5
    generate_page(ml_type, model_dirs, output_filename, class_names, input_filename, title, description, url, block_width, block_height, pixel_scaling, pixel_baseline)


if __name__ == '__main__':
    main()

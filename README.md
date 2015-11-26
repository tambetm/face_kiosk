# Face kiosk

Demo app for testing face recognition models. It captures data from webcam, performs face detection and shows 5 most similar celebrity faces.

## Installation

In Windows install [Anaconda](https://www.continuum.io/downloads) and then follow instructions for installing [precompiled Caffe](http://thirdeyesqueegee.com/deepdream/2015/07/19/running-googles-deep-dream-on-windows-with-or-without-cuda-the-easy-way/). You can use it either with or without CUDA.

In Ubuntu install required Python packages:
```
sudo apt-get install python-opencv python-numpy python-sklearn
```

Finally clone the repository:
```
git clone https://github.com/tambetm/face_kiosk.git
```

## How to run

On Windows run one of these files:

 * `CASIA_lfw.bat` - LFW dataset using model trained on CASIA-WebFace.
 * `lfw.bat` - LFW dataset using model trained on LFW extended with WLF.
 * `fotis.bat` - Fotis dataset using model trained on LFW+WLF+Fotis.
 * `VGG_lfw.bat` - LFW dataset using VGG face model.

**NB!** While the models are included in repository, you still have to download the images. You can download [LFW images](http://vis-www.cs.umass.edu/lfw/) or apply for [CASIA-WebFace images](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html). After downloading the dataset, unpack the images and change the `--images_path` in files above. 

**NB!** VGG face model must be downloaded separately from [here](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/src/vgg_face_caffe.tar.gz). Unpack the file in application folder, which should result in folder `vgg_face_caffe` containing all the required files. VGG face model will benefit a lot from GPU.

## How to prepare a new dataset

Your dataset should be organized into folders, folder name should be person's name (first name and surname separated by underscore) and it should contain cropped face images of this person.

Once you have folder structure in place, you can extract features with following command:

```
python extract.py <images_path> <features.npz> <metadata.csv> <options>
```

First parameter is path to images folder (which has person subfolders), second parameter is name for features file (saved in Numpy .npz format), third parameter is text file name, that will contain file names and person names (saved in CSV format). 

Depending on the model you also need to provide additional options:
 * `--model_file` - Caffe .prototxt file (deploy version)
 * `--pretrained_file` - Caffe .caffemodel file
 * `--mean_file` - Caffe image mean file, either in .binaryproto or .npy format
 * `--image_size` - original image size before cropping, all images are resized to this size
 * `--grayscale` - if input to your network is grayscale image (1 channel)
 * `--oversample` - average features over 10 cropped images (4 corners + center + 2 mirrors of each)
 * `--layer` - name of layer, which features to use
 * `--backend` - either `gpu` or `cpu`
 * `--filter` - which files to look at, e.g. `*.jpg`

Once you have extracted the features in `.npz` file, you need to create index for nearest neighbor search with following command:

```
python build_index.py <features.npz> <index.pkl>
```

Then you need create script or batch file for launching the kiosk with proper options. See the example scripts above.

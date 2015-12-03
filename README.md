# Face kiosk

Demo app for testing face recognition models. It captures data from webcam, performs face detection and shows 5 most similar faces.

## Installation

### Windows prerequisites

In Windows install [Anaconda](https://www.continuum.io/downloads) and then follow instructions for installing [precompiled Caffe](http://thirdeyesqueegee.com/deepdream/2015/07/19/running-googles-deep-dream-on-windows-with-or-without-cuda-the-easy-way/). You can use it either with or without CUDA.

You also need OpenCV, which isn't included with recent Anaconda. First download .whl file from [here](http://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv). Then install it with pip:

```
pip install opencv_python‑2.4.12‑cp27‑none‑win_amd64.whl
```

### Ubuntu prerequisites

In Ubuntu first install Caffe:
```
sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
sudo apt-get install --no-install-recommends libboost-all-dev
sudo apt-get install libatlas-base-dev
sudo apt-get install the python-dev
sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev
git clone https://github.com/BVLC/caffe.git
cd caffe
cp Makefile.config.example Makefile.config
# Adjust Makefile.config (for example, enable USE_CUDNN or CPU_ONLY)
make all
make test
make runtest
```

Then install required Python packages:
```
sudo apt-get install python-opencv python-numpy python-sklearn
```

### Application itself

Just clone the repository:
```
git clone https://github.com/tambetm/face_kiosk.git
```

## How to run

Before you can run the application, you have to download corresponding images, model and data files. If unsure, start with `CASIA_lfw_oversample.[sh|bat]`, which is of reasonable size, runs fast even on CPU and has decent results.

 * `VGG_lfw.[sh|bat]` - LFW dataset using VGG face model. [(images)](http://vis-www.cs.umass.edu/lfw/lfw.tgz) [(model)](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/src/vgg_face_caffe.tar.gz)  [(data)](https://drive.google.com/open?id=0B0fFJSGDUPcgUXpCRXFFMUs4c28)
 * `VGG_lfw_oversample.sh` - LFW dataset using VGG face model with oversampling. [(images)](http://vis-www.cs.umass.edu/lfw/lfw.tgz) [(model)](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/src/vgg_face_caffe.tar.gz)  [(data)](https://drive.google.com/open?id=0B0fFJSGDUPcgUXpCRXFFMUs4c28)
 * `CASIA_lfw.[sh|bat]` - LFW dataset using model trained on CASIA-WebFace. [(images)](http://vis-www.cs.umass.edu/lfw/lfw.tgz) [(model)](https://drive.google.com/open?id=0B0fFJSGDUPcgMVNCYm83T0dyZFk)  [(data)](https://drive.google.com/open?id=0B0fFJSGDUPcgcl92Z0RZVFF1TFE)
 * `CASIA_lfw_oversample.[sh|bat]` - LFW dataset using model trained on CASIA-WebFace with oversampling. [(images)](http://vis-www.cs.umass.edu/lfw/lfw.tgz) [(model)](https://drive.google.com/open?id=0B0fFJSGDUPcgMVNCYm83T0dyZFk)  [(data)](https://drive.google.com/open?id=0B0fFJSGDUPcgcl92Z0RZVFF1TFE)
 * `CASIA.[sh|bat]` - CASIA-WebFace dataset using model trained on CASIA-WebFace. [(images)](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html) [(model)](https://drive.google.com/open?id=0B0fFJSGDUPcgMVNCYm83T0dyZFk)  [(data)](https://drive.google.com/open?id=0B0fFJSGDUPcgZ0owTHNBcE5UUjQ)
 * `CASIA_oversample.[sh|bat]` - CASIA-WebFace dataset using model trained on CASIA-WebFace with oversampling. [(images)](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html) [(model)](https://drive.google.com/open?id=0B0fFJSGDUPcgMVNCYm83T0dyZFk)  [(data)](https://drive.google.com/open?id=0B0fFJSGDUPcgZ0owTHNBcE5UUjQ)
 * `lfw.[sh|bat]` - LFW dataset using model trained on LFW extended with WLF. [(images)](http://vis-www.cs.umass.edu/lfw/lfw.tgz) [(model)](https://drive.google.com/open?id=0B0fFJSGDUPcgTTJSUTNSdmN0aUU)  [(data)](https://drive.google.com/open?id=0B0fFJSGDUPcgUS1wQl9EdVJySnc)
 * `fotis.[sh|bat]` - Fotis dataset using model trained on LFW+WLF+Fotis.  [(model)](https://drive.google.com/open?id=0B0fFJSGDUPcgV0tIaVoxUmRsbW8)  [(data)](https://drive.google.com/open?id=0B0fFJSGDUPcgSUFIaVpDWG5uSXM)

Download the files and unzip them to respective folders - images in `images`, models in `models` and data in `data`. If you already have the images, make symlink in images folder that points to the correct place. Once this is done, run the script.

Oversample means, that features are averaged over 10 cropped images (4 corners + center + 2 mirrors of each).

**NB!** By default Windows scripts use CPU and Linux scripts GPU. This was just my setup, you can change it with `--backend` parameter. VGG face model will benefit a lot from GPU.

## How to prepare a new dataset

Your dataset should be organized into folders, folder name should be person's name (first name and surname separated by underscore) and it should contain **cropped** face images of this person.

Once you have folder structure in place, you can extract features with following command:

```
python src/extract.py <images_path> <features.npz> <metadata.csv> <options>
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

See `extract_CASIA_lfw_oversample.sh` for example. Once you have extracted the features in `.npz` file, you need to create index for nearest neighbor search with following command:

```
python src/build_index.py <features.npz> <index.pkl>
```

Then you need to create a script or batch file for launching the kiosk with proper options. See the example scripts above.

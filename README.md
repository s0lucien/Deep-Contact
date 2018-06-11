# Deep-Contact

This is for master thesis for [Jian Wu](https://github.com/JaggerWu)(xcb479, IT and Cognition, Unversity of Copenhagen), supervised by [Kenny Erleben](http://diku.dk/english/staff/?pure=en/persons/110537)


## Project Description
	
[Project Description](https://github.com/JaggerWu/Deep-Contact/blob/master/Project_description.pdf) for the first draft.
(**Still** Waiting for J.WU to write something here.:cry:)

## Current Task

The task for this week:

  - [x] Install and become familiar with [pybox2d](https://github.com/pybox2d/pybox2d)
  - [x] Generate the data(1000 random samples, 100 objects)

## Experiment

### Command:

Run it with
```
python -m src.random_ball_falling --pause
```

### GIF Examples:

<img src='https://github.com/JaggerWu/Deep-Contact/blob/master/example/nogravity.gif'
     width='40%' height='40%'>
<img src='https://github.com/JaggerWu/Deep-Contact/blob/master/example/normal.gif'
     width='40%' height='40%'>

### XML restore
The we want restore configuration file in XML format and use them for training
afterwards. The configuration file includes bodies and contacts

*Note*:
```
python -m src.gen_data.generate_data -s 30 -p 'path' -n 10
```
to generate the training data

```
<body index="86" type="free">
    <mass value="3.14159274101"/>
    <position x="7.79289388657" y="2.62924313545"/>
    <velocity x="2.7878344059" y="-1.45545887947"/>
    <orientation theta="-0.115291565657"/>
    <inertia value="1.57079637051"/>
    <spin omega="-2.33787894249"/>
    <shape value="circle"/>
</body>
...
<contact index="1" master="2" master_shape="b2CircleShape(childCount=1,
              pos=b2Vec2(0,0),
              radius=1.2000000476837158,
              type=0,
              )" slave="97" slave_shape="b2CircleShape(childCount=1,
              pos=b2Vec2(0,0),
              radius=1.2000000476837158,
              type=0,
              )">
    <position x="0.21963849663734436" y="13.875240325927734"/>
    <normal normal="b2Vec2(-1,2.9819e-05)"/>
    <impulse n="0.005236322991549969" t="-0.002184529323130846"/>
  </contact>
```

Then you can transform the xml data to grid ones which will return to `np.array`

```
python -m src.gen_data.load_xml_save_grid
```

### CNN structure
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_1 (Conv2D)            (None, 41, 41, 64)        2944
_________________________________________________________________
batch_normalization_1 (Batch (None, 41, 41, 64)        256
_________________________________________________________________
activation_1 (Activation)    (None, 41, 41, 64)        0
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 21, 21, 64)        0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 21, 21, 128)       73856
_________________________________________________________________
batch_normalization_2 (Batch (None, 21, 21, 128)       512
_________________________________________________________________
activation_2 (Activation)    (None, 21, 21, 128)       0
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 11, 11, 128)       0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 11, 11, 256)       295168
_________________________________________________________________
batch_normalization_3 (Batch (None, 11, 11, 256)       1024
_________________________________________________________________
activation_3 (Activation)    (None, 11, 11, 256)       0
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 6, 6, 256)         0
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 6, 6, 512)         1180160
_________________________________________________________________
batch_normalization_4 (Batch (None, 6, 6, 512)         2048
_________________________________________________________________
activation_4 (Activation)    (None, 6, 6, 512)         0
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 3, 3, 512)         0
_________________________________________________________________
dropout_1 (Dropout)          (None, 3, 3, 512)         0
_________________________________________________________________
flatten_1 (Flatten)          (None, 4608)              0
_________________________________________________________________
dense_1 (Dense)              (None, 4000)              18436000
_________________________________________________________________
activation_5 (Activation)    (None, 4000)              0
_________________________________________________________________
dropout_2 (Dropout)          (None, 4000)              0
_________________________________________________________________
dense_2 (Dense)              (None, 3362)              13451362
_________________________________________________________________
activation_6 (Activation)    (None, 3362)              0
=================================================================
Total params: 33,443,330
Trainable params: 33,441,410
Non-trainable params: 1,920
_________________________________________________________________
```
Training Config
```
batch_size=1000,
metrics=[losses.mean_absolute_error],
optimizer=optimizers.SGD(
    lr=0.01, decay=1e-6, momentum=0.9, nesterov=True,
),
loss_func=losses.mean_squared_error,
```
Train on `18208` samples, validate on `4553` samples. Training Size
```
Input Size: 41 * 41 *5
Output Size: 3362 * 1
```

Start training by
```
python cnn_training.py -p src/gen_data/data/grid -n 30
```
After training finished, you can check the training logs by
```
tensorboard --logdir ./log/${date_folder}
```

Created by John Gideon 07/27/2019
Based on the paper "Improving Cross-Corpus Speech Emotion Recognition with Adversarial Discriminative Domain Generalization (ADDoG)"
John Gideon, Melvin McInnis, Emily Mower Provost - IEEE Transactions on Affective Computing, 2019



MAIN SCRIPTS

usage: TrainModel.py [-h] [-b BATCHSIZE] [-r REPLEN] [-g GFACTOR]
                     [-d DREPEATS] [-l LEARNINGRATE] [-e MAXEPOCHS] [-v]
                     input model

Train and save a MADDoG model.

positional arguments:
  input                 The path to the input data csv file
  model                 The path to where to save the model

optional arguments:
  -h, --help            show this help message and exit
  -b BATCHSIZE, --batchSize BATCHSIZE
                        The batch size during training (32)
  -r REPLEN, --repLen REPLEN
                        The size of the mid-level representation (128)
  -g GFACTOR, --gFactor GFACTOR
                        The weighting factor for the dataset output loss when
                        training the generator (0.1)
  -d DREPEATS, --dRepeats DREPEATS
                        The amount of repeat batches to train the critic
                        before the generator (5)
  -l LEARNINGRATE, --learningRate LEARNINGRATE
                        The learning rate during training (0.0001)
  -e MAXEPOCHS, --maxEpochs MAXEPOCHS
                        The maximum number of epochs (30)
  -v, --verbose         A flag to print validation UARs (false)
  

  
usage: TestModel.py [-h] [-b BATCHSIZE] input model output

Test a previously trained MADDoG model and output emotion predictions.

positional arguments:
  input                 The path to the input data csv file
  model                 The path to the model directory
  output                The path to where to save the emotion predictions

optional arguments:
  -h, --help            show this help message and exit
  -b BATCHSIZE, --batchSize BATCHSIZE
                        The batch size during testing (32)





SUPPORTING SCRIPTS

DataReader.py: Pytorch dataset and data collating method
Maddog.py: The MADDoG model
Samplers.py: The BatchReplace and SubsetSampler classes for batching data



  

INPUT AND OUTPUT FILE FORMATS

  
Train input format example: train_sample.csv

features,dataset,emotion,validation
/path/to/data/1299.npy,0,1.0;0.0;0.0,0
/path/to/data/65.npy,0,1.0;0.0;0.0,0
/path/to/data/3026.npy,0,,0
/path/to/data/1469.npy,1,0.0;0.0;1.0,1
/path/to/data/75.npy,1,,0
/path/to/data/5286.npy,1,0.0;0.6666666666666666;0.3333333333333333,1
...

features: Path to mfb features for a segment to be used in training
dataset: Unique dataset integer ID between 0 and nDatasets-1
emotion: Semicolon-separated binned values of emotion, blank if no label (3d in this case)
[validation]: Optional input to specify which samples used for validation (only can use labeled data);
              If not given, system will randomly select 1/5 of labeled data
			  
			
			
Model format - directory containing the following files
args.p: Pickle file saving the initializing model parameters (featureLen, emotion and dataset weights, command line args)
G.pt: The trained generator weights
C.pt: The trained classifier weights
D.pt: The trained critic weights



Test input format example: test_sample.csv

features
/path/to/data/3772.npy
/path/to/data/4761.npy
/path/to/data/2971.npy
/path/to/data/6286.npy
/path/to/data/2324.npy
/path/to/data/2804.npy
...

features: Path to mfb features for a segment to be tested



Test output format example: predictions_sample.csv

features,emotion
/path/to/data/3772.npy,-0.07319226;-0.17200315;0.20658702
/path/to/data/4761.npy,-0.13281928;-0.8406483;0.8718629
/path/to/data/2971.npy,-0.060193874;-0.033728;0.07948694
/path/to/data/6286.npy,0.20962527;-1.0837928;0.8565544
/path/to/data/2324.npy,-0.18711832;0.034768783;0.18364501
/path/to/data/2804.npy,-0.00019323453;-0.2876196;0.28731668
...

features: Path to mfb features for the segment that was tested
emotion: The model predicted emotions - semicolon-separated and binned (3d in this case)

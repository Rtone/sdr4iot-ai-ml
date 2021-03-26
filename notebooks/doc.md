# December 2020 :

## Fingerprinting:
- Scene 35-36-37 (3 emitters, 1 emitter for each scene) - 1 server:
-- Various input types:
            whole packet: accuracy around 0,4, lot of overfitting
            various part of the packet: accuracy around 0,4, lot of overfitting
            slices : accuracy around 0,8
            /!\ even though the slices approach performs better it is due to the fact than only the first slice of each packet (i.e corresponding to the header is used) => we now need to try this approach using all slices   
-- Various NN:
CNN1/AlexNet:
 
The best performing version is the tuned one, using the slices approach with slices of 200 samples, and a shift of 50. We also use Gaussian Noise for Data augmentation, as well as BatchNormalization.  


CNN2 :   
   
The best performing version is the tuned one, using the slices approach with slices of 200 samples, and a shift of 50. We also use Gaussian Noise for Data augmentation, as well as BatchNormalization. CNN2 performs a bit worse than CNN1 though.


Conv RNN :
  
The best performing version is the tuned one, using the slices approach with slices of 200 samples, and a shift of 50. We also use Gaussian Noise for Data augmentation, as well as BatchNormalization. Conv RNN performs a bit worse than CNN1 though.   


ResNet 1D:
    
The best performing version is the tuned one, using the slices approach with slices of 200 samples, and a shift of 50. We also use Gaussian Noise for Data augmentation, as well as BatchNormalization. ResNet 1D performs a bit worse than CNN1 though.


CNN Conv2D:
  
The best performing version is the tuned one, using the slices approach with slices of 200 samples, and a shift of 50. We also use Gaussian Noise for Data augmentation, as well as BatchNormalization. CNN Conv2D performs a bit worse than CNN1 though.


CNN1 + Autoencoder:
     
The autoencoder approach does not work very well, even if we tried out various bottlenecks.    
                    

CONCLUSION: The best performing neural network is the tuned CNN1/AlexNet, with an accuracy of 0,84, using the slice input. Nevertheless, we have to keep in mind that only the header of the packet is used, making the classification easier.


- Scene 35-36-37 (3 emitters, 1 emitter for each scene) - 3 servers:
        Using the same NN as with the 1 server situation, the best performing network is still the tuned CNN1/AlexNet, with an accuracy of 0,83, using the slice input with slices of length 500 and a 250 shift. As for the 1 server case, we have to keep in mind that only the header of the packet is used, making the classification easier.


- Scene 31 (all emitters emitting at the same time):
      Using the 1 server case results (tuned CNN1, slice input), this does not work at all: the accuracy is always around 0.4, which is suspicious: maybe the dataset is flawed.
           
           





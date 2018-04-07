"""
Created on Thu Oct 26 14:19:44 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
import cv2
import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import SGD
from torchvision import models
from torch import nn
import pytorchrl as rl
from misc_functions import preprocess_image, recreate_image
import ipdb
class ClassSpecificImageGeneration():
    """
        Produces an image that maximizes a certain class with gradient ascent
    """
    def __init__(self, model,target_class,desired_iterations,file_name,inputImageDims,maximumInternalIterations,mean,std):
        self.mean = mean
        self.std = std
        self.model = model
        self.model.eval()
        self.target_class = target_class
        self.ilr = ILR
        self.iter = desired_iterations
        self.n = file_name
        #Must be in form (channels, height, width)
        self.dims = inputImageDims
        self.maximumInternalIterations = maximumInternalIterations
        # Generate a random image
        self.created_image = np.uint8(np.random.uniform(0, 255, self.dims))
        # Create the folder to export images if not exists
        if not os.path.exists('../generated'):
            os.makedirs('../generated')

    def generate(self):
        initial_learning_rate = self.ilr
        #self.maximumInternalIterations = 3000
        for i in range(1, self.maximumInternalIterations+1):
            if(i%(self.maximumInternalIterations/10) == 0):
                print("Iteration",i)
            # Process image and return variable
            self.processed_image = preprocess_image(self.created_image, mean = self.mean, std = self.std)
            #print("processed_image",self.processed_image.shape)
            """
            im_as_ten = torch.from_numpy(self.created_image.copy()).float()
            im_as_var = Variable(im_as_ten, requires_grad=True)
            self.processed_image = im_as_var
            """
            # Define optimizer for the image]
            optimizer = SGD([self.processed_image], lr=initial_learning_rate, weight_decay=X)
            # Forward pass
            output = self.model(self.processed_image.cuda())
            #output = self.model.forward2(self.processed_image.cuda())
            # Target specific class
            class_loss = -output[0, self.target_class]
            """
            #L1 Norm code.
            l1_crit = nn.L1Loss(size_average=False)
            #ipdb.set_trace()
            reg_loss = 0
            for param in self.model.parameters():
                target = Variable(torch.from_numpy(np.zeros(param.shape).astype(dtype=np.float64))).float()
                reg_loss += l1_crit(param, target.cuda())
            factor = .0005
            class_loss += factor * reg_loss
            """
            # Zero grads
            self.model.zero_grad()
            # Backward
            class_loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = recreate_image(self.processed_image, mean = self.mean, std = self.std)
            #print("created_image",self.created_image.shape)
            # Save image
            if(i == self.maximumInternalIterations):
                #print('Iteration:', str(i), 'Loss', "{0:.2f}".format(class_loss.data.numpy()[0]), " mean:", m)
                cv2.imwrite('../generated/'+ self.n +'_initialLearningrate_' + str(ILR) + '_weightDecay_' + str(X) + '_ClassSpecificImageGeneration_class_'+ str(self.target_class) +'_ir_'+str(self.iter)+'iteration_'+ str(i)+'.jpg', self.created_image[0])
    	
        return self.processed_image

    def divide(arrs,s):
        arr = np.copy(arrs)
        for each in range(arr.shape[0]):
            for z in range(arr.shape[1]):
                arr[each,z] = np.round(arr[each,z]/s)
        return arr
    def add(x1,a4):
        desiredoutputwithCorrectType = np.copy(x1)
        a1 = np.copy(a4)

        for each in range(a1.shape[0]):
            for eac in range(a1.shape[1]):
                desiredoutputwithCorrectType[each,eac] = desiredoutputwithCorrectType[each,eac] + a1[each,eac]
        return desiredoutputwithCorrectType

    def averageImagesAndCombine(n,numOfClasses):
        imagesAveraged = []
        for target_class in range(numOfClasses):
            x = np.zeros(self.dims,dtype = np.uint32)

            for iters in range(1,NUMOFRUNS):
                img = (cv2.imread('../generated/'+ n +'_initialLearningrate_' + str(ILR) + '_weightDecay_' + str(X) + '_ClassSpecificImageGeneration_class_'+ str(target_class) +'_ir_'+str(iters)+'iteration_' + str(self.maximumInternalIterations) + '.jpg',0))

                x = add(x, img)

            print(np.mean(x), np.max(x))

            x = np.divide(x*255,np.max(x))
            print(np.mean(x), np.max(x))

            x = np.copy(x)

            #x = cv2.normalize(x, dst= None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

            cv2.imwrite('../generated/out/'+ n +'_initialLearningrate_' + str(ILR) + '_weightDecay_' + str(X) + '_ClassSpecificImageGeneration_class_'+ str(target_class) +'x_.jpg', x)
            imagesAveraged.append(x)
        return imagesAveraged

if __name__ == '__main__':

     name = 'high_performance.pkl'
     NUMOFRUNS = 5
     ILR = 40
     X = 0
     MII = 3000
     MEAN = [0.485, 0.456, 0.406]
     STD = [0.229, 0.224, 0.225]
     NUMOFCLASSES = 3


     exampleImg = np.zeros((1,100,256))
  
     pretrained_model = torch.load(name)
     for target_class in range(NUMOFCLASSES):
         for iters in range(1,NUMOFRUNS):
             print("Targeting class",target_class,"iter", iters, name)
             csig = ClassSpecificImageGeneration(pretrained_model, target_class, iters, name,exampleImg.shape,MII,MEAN,STD)
             csig.generate()

     csig.averageImagesAndCombine(name)

     name = 'poor_performance.pkl'

     pretrained_model = torch.load(name)
     for target_class in range(NUMOFCLASSES):
         for iters in range(1,NUMOFRUNS):
             print("Targeting class",target_class,"iter", iters, name)
             csig = ClassSpecificImageGeneration(pretrained_model, target_class, iters, name,exampleImg.shape,MII,MEAN,STD)
             csig.generate()
    
     csig.averageImagesAndCombine(name)

     name = 'only_get_boxes_on_right.pkl'

     pretrained_model = torch.load(name)
     for target_class in range(NUMOFCLASSES):
         for iters in range(1,NUMOFRUNS):
             print("Targeting class",target_class,"iter", iters, name)
             csig = ClassSpecificImageGeneration(pretrained_model, target_class, iters, name,exampleImg.shapes,MII,MEAN,STD)
             csig.generate()
    
     csig.averageImagesAndCombine(name)
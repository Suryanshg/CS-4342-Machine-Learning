import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt

# Accuracy Function using the "Percent Correct" Approach
def fPC (y, yhat):
    return np.mean(y==yhat)

# Returns the Ensemble Accuracy of predictors 
def measureAccuracyOfPredictors (predictors, X, y):
    sum_y = np.zeros(y.shape) # Initialize empty array for storing the sum
    for predictor in predictors:
        r1,c1,r2,c2 = predictor
        feature = X[:,r1,c1] - X[:,r2,c2] # Obtain a feature for a particular predictor

        # Performing indicator function a single feature
        feature[feature<=0] = 0 # if value is negative or zero -> 0
        feature[feature>0] = 1 # if value is positive -> 1

        sum_y += feature
    ensemble_y = sum_y / len(predictors)

    # Performing indicator function for the ensemble feature
    ensemble_y[ensemble_y <= 0.5] = 0
    ensemble_y[ensemble_y > 0.5] = 1

    accuracy = fPC(y,ensemble_y)
        
    return accuracy

# Performs Step-wise Classification on the training data for a feature set of 5 elements
def stepwiseRegression (trainingFaces, trainingLabels, testingFaces, testingLabels):
    predictors = [] # List to store the best 5 features in form of tuple of size 4
    for j in range(5):
        bestAccuracy = 0
        bestFeature = (0,0,0,0)
        for r1 in range(24): # Compares all possible set of pixels 
            for c1 in range(24):
                for r2 in range(24):
                    for c2 in range(24):
                        if((r1,c1)!=(r2,c2)): # If both the pixels are different on the basis of location
                            # if((r1,c1,r2,c2) not in predictors): # And the current feature is already not selected as a "Best feature"
                                currentAccuracy = measureAccuracyOfPredictors(predictors + [(r1,c1,r2,c2)], trainingFaces, trainingLabels)
                                if currentAccuracy > bestAccuracy: # if we achieve a best accuracy so far
                                    bestAccuracy = currentAccuracy
                                    bestFeature = (r1,c1,r2,c2) # Set best feature to be the current feature
        predictors.append(bestFeature)
    return predictors

    

# Visualizes the "best" features identified on one image
def visualizeFeatures(predictors, testingFaces):
    
    show = True
    if show:
        # Show an arbitrary test image in grayscale
        im = testingFaces[0,:,:]
        fig,ax = plt.subplots(1)
        ax.imshow(im, cmap='gray')
            
        for predictor in predictors:
            r1,c1,r2,c2 = predictor
            # Show r1,c1
            rect = patches.Rectangle((c1 - 0.5, r1 - 0.5), 1, 1, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            # Show r2,c2
            rect = patches.Rectangle((c2 - 0.5, r2 - 0.5), 1, 1, linewidth=2, edgecolor='b', facecolor='none')
            ax.add_patch(rect)
        
        # Display the merged result
        plt.show()

# Loads data
def loadData (which):
    faces = np.load("{}ingFaces.npy".format(which))
    faces = faces.reshape(-1, 24, 24)  # Reshape from 576 to 24x24
    labels = np.load("{}ingLabels.npy".format(which))
    return faces, labels

# Driver code
if __name__ == "__main__":
    testingFaces, testingLabels = loadData("test")
    trainingFaces, trainingLabels = loadData("train")
    # print(trainingFaces.shape)

    sizes = [400,800,1200,1600,2000]
    # for n in sizes:
    #     predictors = stepwiseRegression(trainingFaces[:n], trainingLabels[:n], testingFaces, testingLabels)
    #     print("Predictors for size "+str(n)+" are "+str(predictors))
    #     print("Training Accuracy:",measureAccuracyOfPredictors(predictors, trainingFaces[:n], trainingLabels[:n]))
    #     print("Testing Accuracy:",measureAccuracyOfPredictors(predictors, testingFaces, testingLabels))
    #     print()

    visualizeFeatures(stepwiseRegression(trainingFaces, trainingLabels, testingFaces, testingLabels), testingFaces)

    # y = np.array([1,0,0,1,1])
    # yhat = np.array([0,0,0,0,1])
    # print(fPC(y,yhat))



import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from copy import deepcopy

# IMPORTANT: I have split the 4 features into two groups(2 and 2) one containing Sepal width and height and the other containing Petal width and height.
# I feel splitting the two and clustering on them seperately makes the most sense and yeilds the most sensible results.
# I have let examples of perfectly clustered runs in the folder as screenshots, since k-means does not cluster perfectly everytime since I am using true random intialization.


#loads the iris.csv file into the program.    
def load_file():
    data = pd.read_csv("iris_data.csv")
    data = pd.DataFrame(data, columns= ['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width'])
    return data

#creates a plot that is clustered using sepal length and width.
def plot_sepal(data, k):
    sepal_length = data['Sepal.Length'].values
    sepal_width = data['Sepal.Width'].values
    
    sepal_list = np.array(list(zip(sepal_length, sepal_width)))

    # X coordinates of the random centroids
    centroid_x = np.random.randint(np.min(sepal_list), np.max(sepal_list), size=k)

    # Y coordinates of random centroids
    centroid_y = np.random.randint(np.min(sepal_list), np.max(sepal_list), size=k)

    centroids_list = np.array(list(zip(centroid_x, centroid_y)), dtype=np.float32)
  
    update(centroids_list, sepal_list, k)

#creates a plot that is clustered using petal length and width.
def plot_petal(data, k):
    petal_length = data['Petal.Length'].values
    petal_width = data['Petal.Width'].values
    
    petal_list = np.array(list(zip(petal_length, petal_width)))

    # X coordinates of the random centroids
    centroid_x = np.random.randint(np.min(petal_list), np.max(petal_list), size=k)

    # Y coordinates of random centroids
    centroid_y = np.random.randint(np.min(petal_list), np.max(petal_list), size=k)

    centroids_list = np.array(list(zip(centroid_x, centroid_y)), dtype=np.float32)
  
    update(centroids_list, petal_list, k, False)

   
#Euclidian Distance calculator 
def euclidian_dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

#main method that assigns and updates. First assigning each point to a centroid and then updating centroids.
#features_list -  the list of features that are being used either sepal length and width or petal length and width.
def update(centroids_list, features_list, k, title = True):
    #Used to store the values of the older centroids.
    old_centroids = np.zeros(centroids_list.shape)
    # Used to update the points and see what cluster they are apart of using 0,1,2 which are classifications stored in this array.
    updated_clusters = np.zeros(len(features_list))
    #Calculates the error distance of old centroid from new centroid.
    distance_error = euclidian_dist(centroids_list, old_centroids, None)

    while distance_error != 0:

        for i in range(len(features_list)):
            #Assign each point to its closest cluster
            distances = euclidian_dist(features_list[i], centroids_list)
            closest_cluster = np.argmin(distances)
            updated_clusters[i] = closest_cluster

        old_centroids = deepcopy(centroids_list)

     
        #taking average value and using it to find new centroids, checking against the updated_cluster array to see which point should effect the centroid.
        for i in range(k):
            temp = []
            for j in range(len(features_list)):
                if updated_clusters[j] == i:
                    temp.append(features_list[j])
            
            if len(temp) != 0:
                centroids_list[i] = np.mean(temp, axis=0)
        distance_error = euclidian_dist(centroids_list, old_centroids, None)
      
    #creating color array for plotting
    color_list = ['r', 'g', 'b']
    fig, ax = plt.subplots()
    # Same as above, check and see what point is realted to what cluster and then plot.
    for i in range(k):
        temp = []
        for j in range(len(features_list)):
                if updated_clusters[j] == i:
                    temp.append(features_list[j])
        if len(temp) != 0:
            for k in range(len(temp)):
                ax.scatter(*temp[k], s=7, c=color_list[i])

    ax.scatter(centroids_list[:, 0], centroids_list[:, 1], marker='+', s=100, c='#050505')

    if title == False:
        plt.title('Petal')
    else:
        plt.title('Sepal')

    plt.show()
    

def main():
    data = load_file()
    #Uncomment one at a time to see fully clustered sepal attributes or petal attributes using 3 centroids for each class. 
    plot_sepal(data, 3)
    #plot_petal(data, 3)


if __name__ == "__main__":
    main()
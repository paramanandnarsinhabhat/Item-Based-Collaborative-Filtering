
#STEP 1 : IMPORTING REQUIRED PACKAGES
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

#STEP 2 : READING DATASET

#Reading ratings file:
ratings = pd.read_csv('/Users/paramanandbhat/Downloads/ImplementationforItemBasedCollaborativeFiltering-201024-234420 (1)/ratings.csv')

#Reading Movie Info File
movie_info = pd.read_csv('/Users/paramanandbhat/Downloads/ImplementationforItemBasedCollaborativeFiltering-201024-234420 (1)/movie_info.csv')

#STEP 3 : MERGE MOVIE INFORMATION TO RATINGS DATAFRAME
'''The movie names are contained
in a separate file.
Let's merge that data with 
ratings and store it in ratings dataframe. 
The idea is to bring movie title information in
ratings dataframe as it would be useful later on'''


ratings = ratings.merge(movie_info[['movie id','movie title']], how='left', left_on = 'movie_id', right_on = 'movie id')

ratings.head()

print(ratings.head())

# STEP 4: COMBINE MOVIE ID AND MOVIE TITLE SEPARATED BY ': ' AND 
#STORE IT IN A NEW COLUMN NAMED MOVIE
ratings['movie'] = ratings['movie_id'].map(str) + str(': ') + ratings['movie title'].map(str)

print(ratings.columns)

# STEP 5: KEEPING THE COLUMNS MOVIE, USER_ID AND 
#RATING IN THE RATINGS DATAFRAME AND DROP ALL OTHERS

ratings = ratings.drop(['movie id', 'movie title', 'movie_id','unix_timestamp'], axis = 1)

print(ratings.columns)

# STEP 6: CREATING TRAIN & TEST DATA & SETTING EVALUATION METRIC

'''In order to test how well we do with 
a given rating prediction method, we would first need
 to define our train and test set, we will only use the train 
 set to build different models and evaluate our model using the
   test set.'''

#Assign X as the original ratings dataframe
X = ratings.copy()


#Split into training and test datasets
X_train, X_test = train_test_split(X, test_size = 0.25, random_state=42)


#Function that computes the root mean squared error (or RMSE)
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


#STEP 7 : SIMPLE BASELINE USING AVERAGE OF ALL RATINGS

#Define the baseline model to always return average of all available ratings
def baseline(user_id, movie):
    return X_train['rating'].mean()

#Function to compute the RMSE score obtained on the test set by a model
def rmse_score(model):
    
    #Construct a list of user-movie tuples from the test dataset
    id_pairs = zip(X_test['user_id'], X_test['movie'])
    
    #Predict the rating for every user-movie tuple
    y_pred = np.array([model(user, movie) for (user, movie) in id_pairs])
    
    #Extract the actual ratings given by the users in the test data
    y_true = np.array(X_test['rating'])
    
    #Return the final RMSE score
    return rmse(y_true, y_pred)


print(rmse_score(baseline))

# STEP 8: ITEM BASED COLLABORATIVE FILTERING WITH SIMPLE ITEM MEAN

'''
Again in item based CF we discussed steps for using 
weighted mean of similar items' ratings, let's first try
just a simple average of all ratings given by a particular
user to all other movies and make predictions. To do that
first we will create the ratings matrix using pandas 
pivot_table function.
'''

# STEP 8.1: BUILD THE RATINGS MATRIX USING PIVOT_TABLE FUNCTION
r_matrix = X_train.pivot_table(values='rating', index='user_id', columns='movie')

print(r_matrix.head())

# STEP 8.2: ITEM BASED COLLABORATIVE FILTER USING MEAN RATINGS
def cf_item_mean(user_id, movie):
        
    #Compute the mean of all the ratings given by the user
    mean_rating = r_matrix.loc[user_id].mean()

    return mean_rating
#Compute RMSE for the Mean model
rmse_score(cf_item_mean)

print(rmse_score(cf_item_mean))

'''
The RMSE score that we get from this simple technique 
is lower than simple user mean that we discussed in the last
module by a small margin, now let us check item based collaborative 
filtering with weighted mean of most similar items
'''
# STEP 9: ITEM BASED COLLABORATIVE FILTERING WITH SIMILARITY WEIGHTED MEAN
'''Now let's use cosine similarity and evaluate item based filtering by using similarity based weighted mean. Now cosine similarity varies from 0 to 1 and the function from sklearn that we are going to use does not work on missing values in the user item matrix so in order to create the item-item matrix we will fill all the missing values with 0. 
This means that for all movie user pairs where we don't have rating will accumulate a 0.'''

#Create a dummy ratings matrix with all null values imputed to 0
r_matrix_dummy = r_matrix.copy().fillna(0)


#Compute the cosine similarity matrix using the dummy ratings matrix
cosine_sim = cosine_similarity(r_matrix_dummy.T, r_matrix_dummy.T)

#Convert into pandas dataframe 
cosine_sim = pd.DataFrame(cosine_sim, index=r_matrix.columns, columns=r_matrix.columns)

cosine_sim.head(5)

print(cosine_sim.head(5))

'''
Using cosine similarity we have estimated the similarity between 
each pair of items and we can use the same to check the most similar
movies to each movie
'''
#Checking movies most similar to Star Wars
cosine_sim['50: Star Wars (1977)'].sort_values(ascending = False)[1:6]


'''Without feeding the information that return of the jedi and empire strikes back belong to the same universe as star wars, we see that cosine similarity has ranked these movies amongst the top. Quite interesting how just the user preferences can be used to find such hidden information.'''


'''Now, we have the item item similarities stored in the matrix cosine_sim. We will define a function to predict the unknown ratings in the test set using item based collarborative filtering with simiarity as cosine and using all the ratings of other items. For each user movie pair:
1. Check if a movie is there in train set, if its not in that case we will just predict the mean rating as the predicted rating
2. Extract cosine similarity values from matrix cosine_sim
3. Drop all the unrated items as they cannot contribute to the prediction from both similarity scores and ratings
4. Use the prediction formula to make rating predictions'''

#Item Based Collaborative Filter using Weighted Mean Ratings
def cf_item_wmean(user_id, movie_id):
    
    #Check if movie_id exists in r_matrix
    if movie_id in r_matrix:
        
        #Get the similarity scores for the item in question with every other item
        sim_scores = cosine_sim[movie_id]
        
        #Get the movie ratings for the user in question
        m_ratings = r_matrix.loc[user_id]
        
        #Extract the indices containing NaN in the m_ratings series
        idx = m_ratings[m_ratings.isnull()].index
        
        #Drop the NaN values from the m_ratings Series (removing unrated items)
        m_ratings = m_ratings.dropna()
        
        #Drop the corresponding cosine scores from the sim_scores series
        sim_scores = sim_scores.drop(idx)
        
        #Compute the final weighted mean
        wmean_rating = np.dot(sim_scores, m_ratings)/ sim_scores.sum()
    
    else:
        #Default to average rating in the absence of any information on the movie in train set
        wmean_rating = X_train['rating'].mean()
    
    return wmean_rating


rmse_score(cf_item_wmean)

print(rmse_score(cf_item_wmean))


#Importing functions to be used in this notebook from Surprise Package
from surprise import Dataset, Reader
from surprise.model_selection import GridSearchCV
from surprise.prediction_algorithms import KNNWithMeans

'''
To load a dataset from a pandas dataframe within Surprise, you will need the load_from_df() method. 
1. You will also need a `Reader` object and the `rating_scale` parameter must be specified. 
2. The dataframe here must have three columns, corresponding to the user (raw) ids, the item (raw) ids, and the ratings in this order. 
3. Each row thus corresponds to a given rating. This is not restrictive as you can reorder the columns of your dataframe easily.
'''
#Reader object to import ratings from X_train
reader = Reader(rating_scale=(1, 5))

#Storing Data in surprise format from X_train
data = Dataset.load_from_df(X_train[['user_id','movie','rating']], reader)



# STEP 10: GRID SEARCH FOR NEIGHBOURHOOD SIZE AND SIMILARITY MEASURE


'''The `cross_validate()` function reports 
accuracy metric over a cross-validation procedure for a 
given set of parameters. If you want to know which parameter
 combination yields the best results, the `GridSearchCV` class 
 comes to the rescue. 

Given a dict of parameters, this class exhaustively tries all 
the combinations of parameters and reports the best parameters for
 any accuracy measure (averaged over the different splits). It is
 heavily inspired from scikit-learn’s GridSearchCV.'''

param_grid = {
    'k': [5, 10, 20],  # list of the number of nearest neighbors to consider
    'sim_options': {
        'name': ['msd', 'cosine', 'pearson'],  # different similarity metrics
        'user_based': [False]  # set to False for item-based CF
    }
}



gs = GridSearchCV(KNNWithMeans, 
                  param_grid, 
                  measures=['rmse'], 
                  cv=5, 
                  n_jobs = -1)

#We fit the grid search on data to find out the best score
gs.fit(data)



#Printing the best score
print(gs.best_score['rmse'])

#Printing the best set of parameters
print(gs.best_params['rmse'])

#Defining similarity measure as per the best parameters
sim_options = {'name': 'cosine', 'user_based': False}

#Fitting the model on train data
model = KNNWithMeans(k = 46, sim_options = sim_options)

#Build full trainset will essentially fits the knnwithmeans on the complete train set instead of a part of it
#like we do in cross validation
model.fit(data.build_full_trainset())

#id pairs for test set
id_pairs = zip(X_test['user_id'], X_test['movie'])

#Making predictions for test set using predict method from Surprise
y_pred = [model.predict(uid = user, iid = movie)[3] for (user, movie) in id_pairs]

#Actual rating values for test set
y_true = X_test['rating']

# Checking performance on test set
rmse(y_true, y_pred)

print(rmse(y_true, y_pred))














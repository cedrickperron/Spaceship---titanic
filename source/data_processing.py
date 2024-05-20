import pandas as pd
import numpy as np


# Data processing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


class DataProcessor(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def processing(self, encoder, col_list):
        """
        Takes as input an encoder (sklearn.preprocessing), X, and the col_list. 

        encoder --- (sklearn.preprocessing): OrdinalEncoder, SimpleImputer, StandardScaler)
        col_list -- list of str

        Returns:
        X -- Processed Data
        """
        X = self[col_list]
        self[col_list] = encoder.fit_transform(X)
        return self
    

    def split_column_and_drop(self, drop_col = []):
        """
        Takes as input the dataset and a list of columns to drop. It automatically splits the Cabin, and drops the
        Passenger_Id and Name columns

        dataset_df - (pandas.Dataframe)
        drop_col - List of str
        """
        ## Split the cabin argument in deck Cabin Number and Side: Cabin Column data are of the form: deck/num/side
        if "Cabin" in self.columns:
            self[["Deck", "Cabin_num", "Side"]] = self["Cabin"].str.split("/", expand = True)
            self = self.drop(["Cabin"], axis = 1)
        
        drop_col = ["PassengerId", "Name"] + drop_col  #Automatically drop the "Passenger_Id", "Name"
        for col in drop_col:
            if col in self.columns:
                self = self.drop([col], axis = 1)

        return self

    def splitting(self, method = "train"):
        """
        Takes as input the dataset and a string that is the type of dataset that we do (training or test). It splits the dataset
        into X and y, and if the method = "Train", it further splits the X and y into a training and testing sample.

        dataset_df - (pandas.Dataframe)
        drop_col - srt
        
        """
        # Check if method is either 'Train' or 'Test'
        if method not in ["train", "test"]:
            raise ValueError("Invalid method. Method must be either 'train' or 'test'.")

        X = self.drop(['Transported'], axis = 1)
        y =self["Transported"]
        if method == "train":
            X_train, X_test, y_train, y_test = train_test_split(X,y,train_size = 0.8,random_state=0)
            return X_train, X_test, y_train, y_test
        else:
            return X, y

    def one_hot_encoding(self, col_names = ["Destination", "Side", "Deck", "HomePlanet"]):
        """
        Takes a dataset (self) and do a one-hot encoding of the data of the col in col_names

        Input:
        self (pandas.Dataframe) -- represents the dataset
        col_names (list of str) --- Name of the col we want to one-hot encode
        """
        for col in col_names:
            if col in self.columns:
                self = pd.get_dummies(self, columns = [col])
        return self

    def run_and_split(self, drop_col = []):
        """
        Full step
        """
        
        self = self.split_column_and_drop(drop_col = drop_col)
        
        # Categorical Encoder
        od = OrdinalEncoder()
        cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Side', 'Deck']
        self  = self.processing(od, col_list = cat_cols)

        # Numerical Encoder
        od = SimpleImputer(strategy="mean")
        num_cols =['Age','FoodCourt','VRDeck','RoomService','Spa', 'ShoppingMall']  #self.select_dtypes(include="number").columns
        self  = self.processing(od, col_list = num_cols)

        # One-hot Encoder
        self = self.one_hot_encoding()

        ## Fill missing values to NaN
        od  = SimpleImputer(strategy = 'constant', fill_value=0)
        col_list = [col for col in self.columns if col != 'Transported']
        self  = self.processing(od, col_list = col_list)

        # Scaling Encoder
        scale = StandardScaler()
        self  = self.processing(scale, col_list = col_list)

        # Splitting
        X_train, X_test, y_train, y_test = self.splitting()
        
        return X_train, X_test, y_train.astype(int), y_test.astype(int)
        



if __name__ == '__main__':
    dataset_df = pd.read_csv("../data/train.csv")
    dataset_df  = DataProcessor(dataset_df)
    
    dataset_df = dataset_df.split_column_and_drop()
    # Categorical Encoder
    od = OrdinalEncoder()
    cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Side', 'Deck']
    dataset_df  = dataset_df.processing(od, col_list = cat_cols)
    
    # Numerical Encoder
    od = SimpleImputer(strategy="mean")
    num_cols = ['Age','FoodCourt','VRDeck','RoomService','Spa', 'ShoppingMall']#dataset_df.select_dtypes(include="number").columns
    dataset_df  = dataset_df.processing(od, col_list = num_cols)

    # One-hot Encoder
    dataset_df = dataset_df.one_hot_encoding()


    ## Fill missing values to NaN
    od  = SimpleImputer(strategy = 'constant', fill_value=0)
    dataset_df  = dataset_df.processing(od, col_list =[col for col in dataset_df.columns if col != 'Transported'])

    print(dataset_df.isna().sum())

    # Scaling Encoder
    scale = StandardScaler()
    dataset_df  = dataset_df.processing(scale, col_list = [col for col in dataset_df.columns if col != 'Transported'])



    
    print("Dataset columns before splitting:", dataset_df.columns)
    print("\n")
    col_drop =  ["Age"]
    dataset_df.split_column_and_drop(col_drop)
    print("Dataset columns after splitting:", dataset_df.columns)
    print("\n")
    print("|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|")
    X_train, X_test, y_train, y_test = dataset_df.splitting()
    print("X_train shape, X_test shape, y_train shape, y_test shape", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    print(y_train)
    import Data_visualisation
    from Data_visualisation import  correlation_matrix
    correlation_matrix(dataset_df)
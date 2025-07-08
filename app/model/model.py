from preprocessing.cleaning_data import Preprocessing
from sklearn.compose import ColumnTransformer
#from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
import joblib
import os


class PolyModel:
    """
    This class create a polynomial model, train and save it
    """

    def __init__(self, save_path="model/best_model.pkl", degree=2):
        self.model = None
        self.degree = degree
        self.save_path = save_path



    def load_data(self):
        print("[INFO] Loading of cleaning data...")
        df = Preprocessing()
        df = df.preprocess()
        if "price" not in df.columns:
            raise ValueError("'price' column missing in the data.")
        return df



    def create_pipeline(self):
        cat_features = ['subtype', 'province']
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_features)
            ],
            remainder='passthrough'
        )

        return make_pipeline(
            preprocessor,
            PolynomialFeatures(degree=self.degree, include_bias=False),
            LinearRegression()
        )



    def train(self):
        if os.path.exists(self.save_path):
            print(f"[SKIP] Model file already exists at '{self.save_path}'. Skipping training.")
            self.model = joblib.load(self.save_path)
            return 

        df = self.load_data()
        X = df.drop(columns=["price"])
        y = df["price"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("[INFO] Training of the model...")
        self.model = self.create_pipeline()
        self.model.fit(X_train, y_train)

        print("[SUCCESS] Model training with success.")

    

    def save(self):
        if self.model is None:
            raise ValueError("Model should be trained before saving.")
        
        joblib.dump(self.model, self.save_path)
        print(f"[SAVE] model save in file '{self.save_path}'.")


if __name__ == "__main__":
    model = PolyModel()
    #model.load_data()
    #model.create_pipeline()
    model.train()
    model.save()
   
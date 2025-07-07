import pandas as pd

class Preprocessing:
    """
    This class upload a csv file from immoweb and clean it of not usefull variable and 
    some empty cell and return a preprocess dataset
    """

    def __init__(self, df):
        self.df = df

    def cleaned_data(self):
        """
        This method receive a csv file, clean and preprocess it
        """
        # Read csv file
        df = pd.read_csv("cleaned_data.csv")  
        df = df.drop(columns=["url", "id"]) # drop columns url, id
        df = df.drop(columns=["MunicipalityCleanName", "locality"]) # same variable as post code
        df = df.drop(columns=["type", "postCode", "hasGarden", "region", "price_square_meter"])
        self.df = df

        return self.df 
    
    def remove_outlier(self):
        """
        This method receive dataFrame and remove outlier
        """

        # Identify unique numerical columns
        numeric_cols = self.df.select_dtypes(include=["float64", "int64"]).columns

        # Delete outliers in each numerical column
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]

        return self.df
    
    def convert_features(self):
        """
        This method convert features like epcScore and buildingCondition
        """

        # convert features epcScore and buildingConstruction to numerical
        epc_score = {"A++": 9,
                    "A+": 8,
                    "A": 7,
                    "B": 6,
                    "C": 5,
                    "D": 4,
                    "E": 3,
                    "F": 2,
                    "G": 1}
        self.df["epc_score"] = self.df["epcScore"].map(epc_score)

        building_condition = {"AS_NEW": 6,
                            "GOOD": 5,
                            "JUST_RENOVATED": 4,
                            "TO_BE_DONE_UP": 3,
                            "TO_RENOVATE": 2,
                            "TO_RESTORE": 1}
        self.df["building_condition"] = self.df["buildingCondition"].map(building_condition)

        self.df = self.df.drop["epcScore", "buildingCondition"] # remove old features

        return self.df
    


task = Preprocessing()
task.cleaned_data()
task.remove_outlier()
task.convert_features()
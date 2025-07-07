import pandas as pd

class Preprocessing:
    """
    This class upload a csv file from immoweb and clean it of not usefull variable and 
    some empty cell and return a preprocess dataset
    """

    def __init__(self, filename = "preprocessing/cleaned_data.csv"):
        
        self.filename = filename

    def preprocess(self):
        """
        This method receive a csv file, clean and preprocess it
        """
        # Read csv file
        df = pd.read_csv(self.filename)  
        df = df.drop(columns=["url", "id"]) # drop columns url, id
        df = df.drop(columns=["MunicipalityCleanName", "locality"]) # same variable as post code
        df = df.drop(columns=["type", "postCode", "hasGarden", "region", "price_square_meter"])
       
        

        # Identify unique numerical columns
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns

        # Delete outliers in each numerical column
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

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
        df["epc_score"] = df["epcScore"].map(epc_score)

        building_condition = {"AS_NEW": 6,
                            "GOOD": 5,
                            "JUST_RENOVATED": 4,
                            "TO_BE_DONE_UP": 3,
                            "TO_RENOVATE": 2,
                            "TO_RESTORE": 1}
        df["building_condition"] = df["buildingCondition"].map(building_condition)
       
        df = df.drop(["epcScore", "buildingCondition"], axis=1) # remove old features

        return df
    


task = Preprocessing()
task.preprocess()
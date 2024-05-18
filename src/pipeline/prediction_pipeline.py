from json import load
import sys
import os
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    def __init__(self) -> None:
        pass

    def predict(self, features):
        try:
            preprocessor_path = os.path.join('artifacts','preprocessor.pkl')
            model_path = os.path.join('artifacts','model.pkl')

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            data_scaled=preprocessor.fit_transform(features)
            pred=model.predict(data_scaled)
            return pred

        except Exception as e:
            logging.info('Exception occured in prediction')
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                age:float,
                sex:str,
                on_thyroxine:str,
                query_on_thyroxine:str,
                on_antithyroid_medication:str,
                sick:str,
                pregnant:str,
                thyroid_surgery:str,
                I131_treatment:str,
                query_hypothyroid:str,
                query_hyperthyroid:str,
                lithium:str,
                goitre:str,
                tumor:str,
                hypopituitary:str,
                psych:str,
                TSH_measured:str,
                TSH:float,
                T3_measured:str,
                T3:float,
                TT4_measured:str,
                TT4:float,
                T4U_measured:str,
                T4U:float,
                FTI_measured:str,
                FTI:float,
                TBG_measured:str,
                referral_source:str
                ):
        self.age = age
        self.sex = sex
        self.on_thyroxine = on_thyroxine
        self.query_on_thyroxine = query_on_thyroxine
        self.on_antithyroid_medication = on_antithyroid_medication
        self.sick = sick
        self.pregnant = pregnant
        self.thyroid_surgery = thyroid_surgery
        self.I131_treatment = I131_treatment
        self.query_hypothyroid = query_hypothyroid
        self.query_hyperthyroid = query_hyperthyroid
        self.lithium = lithium
        self.goitre = goitre
        self.tumor = tumor
        self.hypopituitary = hypopituitary
        self.psych = psych
        self.TSH_measured = TSH_measured
        self.TSH = TSH
        self.T3_measured = T3_measured
        self.T3 = T3
        self.TT4_measured = TT4_measured
        self.TT4 = TT4
        self.T4U_measured = T4U_measured
        self.T4U = T4U
        self.FTI_measured = FTI_measured
        self.FTI = FTI
        self.TBG_measured = TBG_measured
        self.referral_source = referral_source

    
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'age':[self.age ],
                'sex':[self.sex],
                'on_thyroxine':[self.on_thyroxine],
                'query_on_thyroxine':[self.query_on_thyroxine],
                'on_antithyroid_medication':[self.on_antithyroid_medication],
                'sick':[self.sick],
                'pregnant':[self.pregnant],
                'thyroid_surgery':[self.thyroid_surgery],
                'I131_treatment':[self.I131_treatment],
                'query_hypothyroid':[self.query_hypothyroid],
                'query_hyperthyroid':[self.query_hyperthyroid],
                'lithium':[self.lithium],
                'goitre':[self.goitre],
                'tumor':[self.tumor],
                'hypopituitary':[self.hypopituitary],
                'psych':[self.psych],
                'TSH_measured':[self.TSH_measured],
                'TSH':[self.TSH],
                'T3_measured':[self.T3_measured],
                'T3':[self.T3],
                'TT4_measured':[self.TT4_measured],
                'TT4':[self.TT4],
                'T4U_measured':[self.T4U_measured],
                'T4U':[self.T4U],
                'FTI_measured':[self.FTI_measured],
                'FTI':[self.FTI],
                'TBG_measured':[self.TBG_measured],
                'referral_source':[self.referral_source]
            }
            
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df

        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e, sys)

        
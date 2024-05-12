import dataclasses
import os
import sys
import numpy as np
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Data ingestion methods starts')
        try:
            df = pd.read_csv(os.path.join('artifacts','hypothyroid.csv'))
            logging.info('Dataset read as pandas Dataframe')

            df=df.replace({"?":np.NAN})
            logging.info('Replaced NAN')

            df['TSH'] = df['TSH'].str.replace('?','')
            df['TT4'] = df['TT4'].str.replace('?','')
            df['T4U'] = df['T4U'].str.replace('?','')
            df['FTI'] = df['FTI'].str.replace('?','')
            df['age'] = df['age'].str.replace('?','')

            df.drop(columns='TBG', inplace=True)
            df['sex'] = df['sex'].str.replace('?','')
            logging.info('stage 2 in EDA')

            df['age'] = pd.to_numeric(df['age'], errors='coerce')
            df['TSH'] = pd.to_numeric(df['TSH'], errors='coerce')
            df['T3'] = pd.to_numeric(df['T3'], errors='coerce')
            df['TT4'] = pd.to_numeric(df['TT4'], errors='coerce')
            df['T4U'] = pd.to_numeric(df['T4U'], errors='coerce')
            df['FTI'] = pd.to_numeric(df['FTI'], errors='coerce')

            condition = df['age'] > 100
            df.drop(df[condition].index, inplace=True)
            logging.info('Condition passed')

            df['sex']=df['sex'].fillna(df['sex'].mode().iloc[0])
            df['TSH'].fillna(df['TSH'].median(), inplace=True)
            df['T3'].fillna(df['T3'].median(), inplace=True)
            df['TT4'].fillna(df['TT4'].median(), inplace=True)
            df['T4U'].fillna(df['T4U'].median(), inplace=True)
            df['FTI'].fillna(df['FTI'].median(), inplace=True)

            df['age'].fillna(df['age'].median(), inplace=True)
             
            df.rename(columns={'on thyroxine': 'on_thyroxine', 'query on thyroxine': 'query_on_thyroxine','on antithyroid medication':'on_antithyroid_medication','thyroid surgery':'thyroid_surgery','I131 treatment':'I131_treatment','query hypothyroid':'query_hypothyroid','query hyperthyroid':'query_hyperthyroid','TSH measured':'TSH_measured','T3 measured':'T3_measured','TT4 measured':'TT4_measured','T4U measured':'T4U_measured','FTI measured':'FTI_measured','TBG measured':'TBG_measured','referral source':'referral_source'}, inplace=True)
            logging.info("EDA completed performing ingestion")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info('Train Test Split')

            train_set, test_set = train_test_split(df, test_size=20)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)
            logging.info('Ingestion of data completed')

            return (self.ingestion_config.train_data_path,
                    self.ingestion_config.test_data_path
                    
                )

        except Exception as e:
            logging.info('Exception occured at data ingestion stage')
            raise CustomException(e, sys)


if __name__=='__main__':
    obj=DataIngestion()
    train_data_path,test_data_path=obj.initiate_data_ingestion()
    
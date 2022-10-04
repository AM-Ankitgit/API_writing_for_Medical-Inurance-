import pickle
import json
import config
import numpy as np
##  Note class for user inputs

class MedicalInsurance():
    #create method for  user inputs
    def __init__(self,age,sex,bmi,children,smoker,region):
        self.age=age
        self.sex=sex
        self.bmi=bmi
        self.children=children
        self.smoker=smoker
        self.region="region_"+region
    # create for Laod model

    def load_model(self):
        with open(config.MODEL_FIle_PATH,"rb") as file:
            self.model=pickle.load(file)
        with open(config.JSON_File_Path,"r") as file:
            self.json_data=json.load(file)

    ## pass CReate api
    def get_predicted_charges(self):

        self.load_model()             ## this inbuilt function
        
        # create test array

        test_array = np.zeros(len(self.json_data['columns'])) ## column is of json file
        test_array[0]=self.age
        test_array[1]=self.json_data['sex'][self.sex]
        test_array[2]=self.bmi
        test_array[3]= self.children

        test_array[4]= self.json_data['smoker'][self.smoker]

        region_index =self.json_data['columns'].index(self.region)
        
        test_array[region_index] = 1
        print("Test_Array>>",test_array)
        predict_charges=self.model.predict([test_array])
        return predict_charges



    
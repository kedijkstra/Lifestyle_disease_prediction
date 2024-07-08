import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import datetime
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import scipy.stats as sps
import warnings

__author__ = "Keimpe Dijkstra"
__credits__ = ["Stefan Wijtsma"]
__license__ = "GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007"
__version__ = "1.0.1"
__maintainer__ = "Keimpe Dijkstra"
__email__ = "k.dijkstra@labonovum.com"

class dataPreprocessing():
    '''This class handles the preprocessing of selected files obtained from the UK biobank
    '''

    def __init__(self, wd) :
        #Data
        self.family_history = wd + "NHS/Data_files/Grouped_files/replace/Family_history.csv"
        self.attendance = wd + "NHS/Data_files/Grouped_files/replace/Dates_attending_assessment_centers_participant.csv"
        self.demographics = wd + "NHS/Data_files/Grouped_files/replace/Demographics.csv"
        self.bodymeasures = wd + "NHS/Data_files/Grouped_files/replace/Body_measures.csv"
        self.blood_biomarkers = wd + "NHS/Data_files/Grouped_files/replace/Blood_biomarkers.csv"
        self.bloodpressure = wd + "NHS/Data_files/Grouped_files/raw/Blood_pressure_raw.csv"
        self.urine_biomarkers = wd + "NHS/Data_files/Grouped_files/replace/Urine_biomarkers.csv"
        self.medical_conditions = wd + "NHS/Data_files/Grouped_files/replace/Medical_conditions.csv"
        self.alcohol = wd + "NHS/Data_files/Grouped_files/replace/Alcohol.csv"
        self.physical_activity = wd + "NHS/Data_files/Grouped_files/replace/Physical_activity.csv"
        self.sleep = wd + "NHS/Data_files/Grouped_files/replace/Sleep.csv"
        self.smoking = wd + "NHS/Data_files/Grouped_files/replace/Smoking.csv"
        self.smoking_supplementary = wd + "NHS/Data_files/supplementary_data/smokers_data_keimpe_participant.csv"
        self.white_bloodcell = wd + "NHS/Data_files/Separate_files/coding_option_replace/Blood_biomarkers_3.csv"
        self.symptomes = wd + "NHS/Data_files/Grouped_files/replace/Symptoms_and_pain.csv"
        
        #First occurences
        self.first_occurence_copd = wd + "NHS/Data_files/Labeling/copd_first_occurence_dates.csv"
        self.first_occurence_asthma = wd + "NHS/Data_files/Labeling/asthma_first_occurence_dates.csv"
        self.first_occurence_diabetes = wd + "NHS/Data_files/Labeling/DM_first_occurence_dates.csv"
        self.first_occurence_osteoporosis = wd + "NHS/Data_files/Labeling/osteoporosis_diagnosis_dates.csv"
        self.first_occurence_cvd = wd + "NHS/Data_files/supplementary_data/CVD_first_occurrences_with_labels_participant.csv"

        #Coding
        self.coding819 = wd + "NHS/Data_files/coding/coding819.tsv"

        #Variables
        self.meaning = None
        self.coding = None
        self.df = None
        self.comorbidities = []


    def merge_noneoftheabove(self, df):
        '''This functions replaces none of the above from group 1 and 2 into general none of the above
        '''
        return df.replace(to_replace="None of the above (group 1)|None of the above (group 2)", value="None of the above")
    
    def remove_redundant_noneoftheabove(self, df):
        '''This function replaces none of the above from group 1 and 2 and replaces them with an empty string
        '''
        df =  df.replace(to_replace="\|None of the above \(group 1\)", value="", regex=True)
        df =  df.replace(to_replace="\|None of the above \(group 2\)", value="", regex=True)
        return df
    
    def diagnosis_including_diabetes(self, df, df_columns):
        '''
        Consructs three lists:
         1) dm: The various sets of diseases containing dm
         2) other: The various sets of diseases not containing dm
         1) uncertain: The condition of familymember is not specified

        Parameters
        ----------
        df : pandas dataframe object
            Dataframe containing data to be used
        df_columns : List
            List with column names as string to be evaluated

        Returns
        -------
        dm : list
        other : list
        uncertain : list
        '''
        all = []
        for col in df_columns:
            for i in df[col].unique():
                if i not in all:
                    all.append(i)
                    
        dm = []
        other = []
        uncertain = []
        for i in all:
            if type(i) == str:
                if re.search("Diabetes", i):
                    dm.append(i)
                elif re.search("Do not know", i):
                    uncertain.append(i)
                else:
                    other.append(i)
            else:
                uncertain.append(i)

        return dm, other, uncertain

    def pre_family_history(self):
        '''
        Returns dataframe with three rows indicating the presence of diabetes in father, mother and sibling binary

        Returns
        -------
        df : pandas dataframe object
            Dataframe with diabetes values for different family members
        '''
        fh = pd.read_csv(self.family_history)
        fh = self.merge_noneoftheabove(fh)
        fh = self.remove_redundant_noneoftheabove(fh)
        dm, other, uncertain = self.diagnosis_including_diabetes(fh, fh.columns[1:])
        #Create new columns indicating whether a familymember suffers from diabetes (1:yes,0:no,3:uncertain)
        fh.loc[:, ("Illnesses of father | Instance 0",
                "Illnesses of father | Instance 1",
                "Illnesses of father | Instance 2",
                "Illnesses of father | Instance 3",
                "Illnesses of mother | Instance 0",
                "Illnesses of mother | Instance 1",
                "Illnesses of mother | Instance 2",
                "Illnesses of mother | Instance 3",
                "Illnesses of siblings | Instance 0",
                "Illnesses of siblings | Instance 1",
                "Illnesses of siblings | Instance 2",
                "Illnesses of siblings | Instance 3"
                )] = fh.loc[:, ("Illnesses of father | Instance 0",
                "Illnesses of father | Instance 1",
                "Illnesses of father | Instance 2",
                "Illnesses of father | Instance 3",
                "Illnesses of mother | Instance 0",
                "Illnesses of mother | Instance 1",
                "Illnesses of mother | Instance 2",
                "Illnesses of mother | Instance 3",
                "Illnesses of siblings | Instance 0",
                "Illnesses of siblings | Instance 1",
                "Illnesses of siblings | Instance 2",
                "Illnesses of siblings | Instance 3"
                )].replace(to_replace=dm, value=1)

        fh.loc[:, ("Illnesses of father | Instance 0",
                "Illnesses of father | Instance 1",
                "Illnesses of father | Instance 2",
                "Illnesses of father | Instance 3",
                "Illnesses of mother | Instance 0",
                "Illnesses of mother | Instance 1",
                "Illnesses of mother | Instance 2",
                "Illnesses of mother | Instance 3",
                "Illnesses of siblings | Instance 0",
                "Illnesses of siblings | Instance 1",
                "Illnesses of siblings | Instance 2",
                "Illnesses of siblings | Instance 3"
                )] = fh.loc[:, ("Illnesses of father | Instance 0",
                "Illnesses of father | Instance 1",
                "Illnesses of father | Instance 2",
                "Illnesses of father | Instance 3",
                "Illnesses of mother | Instance 0",
                "Illnesses of mother | Instance 1",
                "Illnesses of mother | Instance 2",
                "Illnesses of mother | Instance 3",
                "Illnesses of siblings | Instance 0",
                "Illnesses of siblings | Instance 1",
                "Illnesses of siblings | Instance 2",
                "Illnesses of siblings | Instance 3"
                )].replace(to_replace=other, value=0)

        fh.loc[:, ("Illnesses of father | Instance 0",
                "Illnesses of father | Instance 1",
                "Illnesses of father | Instance 2",
                "Illnesses of father | Instance 3",
                "Illnesses of mother | Instance 0",
                "Illnesses of mother | Instance 1",
                "Illnesses of mother | Instance 2",
                "Illnesses of mother | Instance 3",
                "Illnesses of siblings | Instance 0",
                "Illnesses of siblings | Instance 1",
                "Illnesses of siblings | Instance 2",
                "Illnesses of siblings | Instance 3"
                )] = fh.loc[:, ("Illnesses of father | Instance 0",
                "Illnesses of father | Instance 1",
                "Illnesses of father | Instance 2",
                "Illnesses of father | Instance 3",
                "Illnesses of mother | Instance 0",
                "Illnesses of mother | Instance 1",
                "Illnesses of mother | Instance 2",
                "Illnesses of mother | Instance 3",
                "Illnesses of siblings | Instance 0",
                "Illnesses of siblings | Instance 1",
                "Illnesses of siblings | Instance 2",
                "Illnesses of siblings | Instance 3"
                )].replace(to_replace=uncertain, value=3)

        conditions = [ (fh["Illnesses of father | Instance 0"] == 1),
                    (fh["Illnesses of father | Instance 1"] == 1),
                    (fh["Illnesses of father | Instance 2"] == 1),
                    (fh["Illnesses of father | Instance 3"] == 1),
                    (fh["Illnesses of father | Instance 0"] == 3),
                    (fh["Illnesses of father | Instance 1"] == 3),
                    (fh["Illnesses of father | Instance 2"] == 3),
                    (fh["Illnesses of father | Instance 3"] == 3),
                    (fh["Illnesses of father | Instance 0"] == 0),
                    (fh["Illnesses of father | Instance 1"] == 0),
                    (fh["Illnesses of father | Instance 2"] == 0),
                    (fh["Illnesses of father | Instance 3"] == 0)
                    ]

        values = [1,1,1,1,3,3,3,3,0,0,0,0]
        fh["Illnesses of father"] = np.select(conditions, values)

        conditions = [
                    (fh["Illnesses of mother | Instance 0"] == 1),
                    (fh["Illnesses of mother | Instance 1"] == 1),
                    (fh["Illnesses of mother | Instance 2"] == 1),
                    (fh["Illnesses of mother | Instance 3"] == 1),
                    (fh["Illnesses of mother | Instance 0"] == 3),
                    (fh["Illnesses of mother | Instance 1"] == 3),
                    (fh["Illnesses of mother | Instance 2"] == 3),
                    (fh["Illnesses of mother | Instance 3"] == 3),
                    (fh["Illnesses of mother | Instance 0"] == 0),
                    (fh["Illnesses of mother | Instance 1"] == 0),
                    (fh["Illnesses of mother | Instance 2"] == 0),
                    (fh["Illnesses of mother | Instance 3"] == 0)]
        fh["Illnesses of mother"] = np.select(conditions, values)

        conditions = [
                    (fh["Illnesses of siblings | Instance 0"] == 1),
                    (fh["Illnesses of siblings | Instance 1"] == 1),
                    (fh["Illnesses of siblings | Instance 2"] == 1),
                    (fh["Illnesses of siblings | Instance 3"] == 1),
                    (fh["Illnesses of siblings | Instance 0"] == 3),
                    (fh["Illnesses of siblings | Instance 1"] == 3),
                    (fh["Illnesses of siblings | Instance 2"] == 3),
                    (fh["Illnesses of siblings | Instance 3"] == 3),
                    (fh["Illnesses of siblings | Instance 0"] == 0),
                    (fh["Illnesses of siblings | Instance 1"] == 0),
                    (fh["Illnesses of siblings | Instance 2"] == 0),
                    (fh["Illnesses of siblings | Instance 3"] == 0)]
        fh["Illnesses of siblings"] = np.select(conditions, values)

        fh = fh.drop(fh.columns[1:13], axis=1)
        fh = fh[1:].replace(3,0)
        return fh
    
    def ethnicity_regrouping(self, df, col_name):  # demographics
        '''
        This function regroups the different ethinicities into bigger groups as
        provided by the ethnicity_mapping dictionary.

        Parameters
        ----------
        df : pandas dataframe object
            Dataframe containing data to be used
        col_name : String
            Name of column containing ethnicity values

        Returns
        -------
        df : pandas dataframe object
            Dataframe with grouped ethnicities
        '''
        try:
                if "Ethnic background" not in col_name:
                    raise ValueError("Double check that the column is the ethnicity background column" )
            
        except ValueError as e:
            print("Error:", e)
            # You can choose to re-raise the exception to propagate it to the calling code
            raise e    
        else:
            ethnicity_mapping = {    # We use the same groups as explained in the AMS.
                'British': 'White',          
                'Irish': 'White',
                'White' : 'White',
                'Any other white background': 'White',
                        
                'Mixed': 'Mixed',
                'White and Black Caribbean': 'Mixed',
                'White and Black African': 'Mixed',
                'White and Asian': 'Mixed',
                'Any other mixed background': 'Mixed',    
                
                'Asian or Asian British': 'Asian',
                'Indian' : 'Asian',
                'Pakistani' : 'Asian',
                'Bangladeshi':  'Asian',
                'Any other Asian background': 'Asian',
                
                'Black or Black British' : 'Black',
                'African' : 'Black',
                'Caribbean': 'Black',
                'Any other Black background': 'Black',  
                
                'Chinese': 'Chinese',
                'Other ethnic group': 'Other'
                                
            }

            # Replace the values in the 'ethnicity' column using the mapping
            df[col_name] = df[col_name].map(ethnicity_mapping)

            return df
        
    def pre_demographics_ethnicity(self):
        '''
        Returns a dataframe with one hot encodings for six main ethnicities

        Returns
        -------
        df : pandas dataframe object
        '''
        d = pd.read_csv(self.demographics)
        d = self.ethnicity_regrouping(d, "Ethnic background | Instance 0")
        d = d[["Participant ID", "Ethnic background | Instance 0"]]
        one_hot = pd.get_dummies(d["Ethnic background | Instance 0"])
        d = pd.concat([d, one_hot], axis=1)
        d = d[['Participant ID',
               'Asian',
                'Black',
                'Chinese',
                'Mixed',
                'Other',
                'White']]
        return d
    
    def pre_demographics_basic(self):
        '''
        Returns a dataframe with age and sex columns

        Returns
        -------
        df : pandas dataframe object
        '''
        d = pd.read_csv(self.demographics)
        d = d[['Participant ID', 'Age at recruitment', 'Sex']]
        d["Sex"] = d[["Sex"]].apply(self.sex_to_binary, axis=1)
        return d
    
    def pre_bodymeasures(self):
        '''
        Returns a dataframe with various columns on different type of body measures 

        Returns
        -------
        df : pandas dataframe object
        '''
        bm = pd.read_csv(self.bodymeasures)
        bm["fmi"] = bm["Whole body fat mass | Instance 0"] / ((bm["Standing height | Instance 0"] /100)* (bm["Standing height | Instance 0"]/100))#Create fat mass index column
        bm = bm[["Participant ID","fmi",
                "Body mass index (BMI) | Instance 0",
                "Body fat percentage | Instance 0",
                "Waist circumference | Instance 0",
                "Weight | Instance 0",
                "Hip circumference | Instance 0",
                "Whole body fat mass | Instance 0",
                "Basal metabolic rate | Instance 0",
                "Trunk fat percentage | Instance 0",
                "Arm fat percentage (left) | Instance 0",
                "Leg fat percentage (left) | Instance 0",
                'Standing height | Instance 0'
                ]]
        return bm
    
    def sex_to_binary(self, i):
        '''Replaces a variable indicating gender by binary numbers

        Parameters
        -------
        i : String
            Either Female or Male
        
        Returns
        -------
        i : Integer
            binary encoding for gender
        '''
        if  re.search("Female",str(i)):
            return 0
        if re.search("Male", str(i)):
            return 1
        else:
            return i
    
    def pre_blood_biomarker(self, dem_basic):
        '''
        Returns dataframe with blood biomarker columns 
        
        Returns
        -------
        df : pandas dataframe object
        '''
        bbm = pd.read_csv(self.blood_biomarkers)
        first_and_general_instances = [i for i in list(bbm.columns) if not re.search("Instance 1|Instance 2|Instance 3", i)]
        remove_columns = [ i for i in bbm.columns if i not in first_and_general_instances ]
        bbm = bbm.drop(remove_columns, axis=1)
        bbm["trigl_hdl_ratio"] = (bbm["Triglycerides | Instance 0"] * 88.57 ) / (bbm["HDL cholesterol | Instance 0"] *38.67)
        bbm["apob_apoa_ratio"] = bbm["Apolipoprotein B | Instance 0"] / bbm["Apolipoprotein A | Instance 0"]
        bbm = bbm.drop(columns=['Glycated haemoglobin (HbA1c) assay date | Instance 0'])
        bbm = bbm.merge(dem_basic , on="Participant ID") 
        return bbm

    def pre_blood_pressure(self):
        '''
        Returns a dataframe with diastolic and systolic blood pressure

        Returns
        -------
        df : pandas dataframe object
        '''
        bp = pd.read_csv(self.bloodpressure)

        bp['Diastolic blood pressure'] = bp[['Diastolic blood pressure, automated reading | Instance 0 | Array 0',
        'Diastolic blood pressure, automated reading | Instance 0 | Array 1']].mean(axis=1)
        bp['Systolic blood pressure'] = bp[['Systolic blood pressure, automated reading | Instance 0 | Array 0',
        'Systolic blood pressure, automated reading | Instance 0 | Array 1']].mean(axis=1)

        return bp[["Participant ID",'Diastolic blood pressure', 'Systolic blood pressure' ]]
    
    def pre_urine_biomarkers(self):
        '''
        Returns a dataframe with urine biomarkers

        Returns
        -------
        df : pandas dataframe object
        '''
        bm = pd.read_csv(self.urine_biomarkers, low_memory=False)
        first_and_general_instances = [i for i in list(bm.columns) if not re.search("Instance 1|Instance 2|Instance 3|flag", i)]
        remove_columns = [ i for i in bm.columns if i not in first_and_general_instances ]
        bm = bm.drop(remove_columns, axis=1)
        bm["Creatinine (enzymatic) in urine | Instance 0"] = bm["Creatinine (enzymatic) in urine | Instance 0"] / 1000000 * 113.12 #convert creatine in urine from mmol/L to mg/dL
        bm["albumin_creatine_ratio"] = bm["Microalbumin in urine | Instance 0"] / bm["Creatinine (enzymatic) in urine | Instance 0"] #add albumine creatine ratio
        return bm
    

    def pre_medical_conditions(self):
        '''
        Function which creates da datframe for medical conditions

        Returns
        -------
        df : pandas dataframe object
        '''
        mc = pd.read_csv(self.medical_conditions, low_memory=False)
        
        #Comorbilities
        mc['Non-cancer illness code, self-reported | Instance 0'].apply(self.get_comorb)
        for i in self.comorbidities:
            mc[i] = mc['Non-cancer illness code, self-reported | Instance 0'].apply(self.mc_search, current_comord=i)

        #Medication
        mc["Cholesterol_lowering_medication"] = mc[["Medication for cholesterol, blood pressure or diabetes | Instance 0", 'Medication for cholesterol, blood pressure, diabetes, or take exogenous hormones | Instance 0']].apply(self.str_check_twocolumn, string="Cholesterol lowering medication",axis=1)
        mc["Insulin"] = mc[["Medication for cholesterol, blood pressure or diabetes | Instance 0", 'Medication for cholesterol, blood pressure, diabetes, or take exogenous hormones | Instance 0']].apply(self.str_check_twocolumn, string="Insulin", axis=1)
        mc["Blood_pressure"] = mc[["Medication for cholesterol, blood pressure or diabetes | Instance 0", 'Medication for cholesterol, blood pressure, diabetes, or take exogenous hormones | Instance 0']].apply(self.str_check_twocolumn, string="Blood pressure medication", axis=1)
        mc['Doctor diagnosed asthma'] = mc['Doctor diagnosed asthma'].apply(self.yes_no_binary)
        mc['Diabetes diagnosed by doctor | Instance 0'] = mc['Diabetes diagnosed by doctor | Instance 0'].replace("Prefer not to answer", pd.np.nan)
        mc['Diabetes diagnosed by doctor | Instance 0'] = mc['Diabetes diagnosed by doctor | Instance 0'].replace("Do not know", pd.np.nan)
        mc['Diabetes diagnosed by doctor | Instance 0'] =mc['Diabetes diagnosed by doctor | Instance 0'].apply(self.yes_no_binary)

        #diagnosed by doctor
        mc["Hayfever_allergic_rhinitis_eczema_doctor"] = mc[['Blood clot, DVT, bronchitis, emphysema, asthma, rhinitis, eczema, allergy diagnosed by doctor | Instance 0']].apply(self.Hayfever_allergic_rhinitis_eczema_doctor_check, axis=1)
        mc["Emphysema_chronic_bronchitis_doctor"] = mc[['Blood clot, DVT, bronchitis, emphysema, asthma, rhinitis, eczema, allergy diagnosed by doctor | Instance 0']].apply(self.Emphysema_chronic_bronchitis_doctor_check, axis=1)
        
        #Binairy to ones and zeros
        mc["Doctor diagnosed bronchiectasis"] = mc[['Doctor diagnosed bronchiectasis']].apply(self.yes_no_binary, axis=1)
        mc["Doctor diagnosed hayfever or allergic rhinitis"] = mc[['Doctor diagnosed hayfever or allergic rhinitis']].apply(self.Emphysema_chronic_bronchitis_doctor_check, axis=1)
        mc["Doctor diagnosed chronic bronchitis"] = mc[['Doctor diagnosed chronic bronchitis']].apply(self.yes_no_binary, axis=1)
        mc["Doctor diagnosed hayfever or allergic rhinitis"] = mc[['Doctor diagnosed hayfever or allergic rhinitis']].apply(self.Emphysema_chronic_bronchitis_doctor_check, axis=1)

        return mc
    
    def Hayfever_allergic_rhinitis_eczema_doctor_check(self, i):
        if re.search("Hayfever, allergic rhinitis or eczema", str(i)):
                return 1
        if re.search("nan", str(i)):
                return pd.np.nan
        return 0
    
    def Emphysema_chronic_bronchitis_doctor_check(self, i):
        if re.search("Emphysema/chronic bronchitis", str(i)):
                return 1
        if re.search("nan", str(i)):
                return pd.np.nan
        return 0
    
    def yes_no_binary(self, i):
        '''
        This function replaces yes and no into binary values

        Parameters
        -------
        i : String
            Either Yes or no
        
        Returns
        -------
        i : Integer
            binary encoding for option
        '''
        if str(i) == "Yes":
            return 1
        return 0

    def get_comorb(self, i):
        '''
        Returns list with different obseverd comorbidities

        Parameters
        -------
        i : String


        Returns
        -------
        list : list
            List containing applicable comborbidities
        '''
        if str(i) != "nan":
            i = i.split("|")
            for j in i:
                if j not in self.comorbidities:
                    self.comorbidities.append(j)

    def mc_search(self, instance_comord, current_comord):
        if re.search(current_comord, str(instance_comord)):
            return 1
        return 0
    
    def str_check_twocolumn(self, i, string):
        if re.search(string, str(i[0])) or re.search(string, str(i[1])):
            return 1
        return 0
        
    def pre_alcohol(self): 
        '''
        Returns one hot encoding for selected rates of alcohol consumption

        Returns
        -------
        df : pandas dataframe object
        '''
        alc = pd.read_csv(self.alcohol, sep=",", low_memory=False)
        alc = pd.get_dummies(alc, columns=["Alcohol intake frequency. | Instance 0"])
        alc = alc[['Participant ID', 
            'Alcohol intake frequency. | Instance 0_Daily or almost daily',
            'Alcohol intake frequency. | Instance 0_Never',
            'Alcohol intake frequency. | Instance 0_Once or twice a week',
            'Alcohol intake frequency. | Instance 0_One to three times a month',
            'Alcohol intake frequency. | Instance 0_Special occasions only',
            'Alcohol intake frequency. | Instance 0_Three or four times a week']]
        return alc
    
    def pre_physical_activity(self):
        '''
        Returns selected physical activity measures

        Returns
        -------
        df : pandas dataframe object
        '''
        pa = pd.read_csv(self.physical_activity, low_memory=False)
        one_hot = pd.get_dummies(pa['Usual walking pace | Instance 0'])
        pa = pd.concat([pa, one_hot], axis=1)
        return pa
    
    def pre_sleep(self):
        '''
        Returns selected sleep measures

        Returns
        -------
        df : pandas dataframe object
        '''
        sleep = pd.read_csv(self.sleep, low_memory=False)
        sleep = sleep[["Participant ID", "Sleep duration | Instance 0"]]
        sleep = sleep.replace("Prefer not to answer", pd.np.nan)
        sleep = sleep.replace("Do not know", pd.np.nan)
        return sleep
    
    def pre_smoking(self):
        '''
        Returns dataframe with columns indicating if a participant smokes, or has smoked

        Returns
        -------
        df : pandas dataframe object
        '''
        smoking = pd.read_csv(self.smoking, low_memory=False)
        smoking = smoking[["Participant ID","Tobacco smoking"]]
        smoking = pd.get_dummies(smoking, columns=["Tobacco smoking", ])
        smoking = smoking[['Participant ID', 'Tobacco smoking_Ex-smoker',
                            'Tobacco smoking_Never smoked', 'Tobacco smoking_Occasionally',
                            'Tobacco smoking_Smokes on most or all days']]
        return smoking
    
    def pre_smoking_supplementary(self):
        '''
        Creates dataframe with extensive smoking data

        Returns
        -------
        df : pandas dataframe object
        '''
        smoking_data = pd.read_csv(self.smoking_supplementary)
        smoking_data = smoking_data.rename(columns={"eid": 'Participant ID', "p20161_i0":"Pack years of smoking",
                                          "p20162_i0":"Pack years adult smoking as proportion of life span exposed to smoking",
                                          "p3436_i0":"Age started smoking in current smokers",
                                          "p2867_i0":"Age started smoking in former smokers",
                                          "p3456_i0":"Number of cigarettes currently smoked daily (current cigarette smokers)",
                                          "p6183_i0":"Number of cigarettes previously smoked daily (current cigar/pipe smokers)",
                                          "p2887_i0":"Number of cigarettes previously smoked daily"})
        smoking_data['Age started smoking in current smokers'] = smoking_data['Age started smoking in current smokers'].replace("Do not know", pd.np.nan)
        smoking_data['Age started smoking in current smokers'] = smoking_data['Age started smoking in current smokers'].replace("Prefer not to answer", pd.np.nan)
        smoking_data["Age started smoking in current smokers"] = pd.to_numeric(smoking_data["Age started smoking in current smokers"])

        smoking_data['Age started smoking in former smokers'] = smoking_data['Age started smoking in former smokers'].replace("Do not know", pd.np.nan)
        smoking_data['Age started smoking in former smokers'] = smoking_data['Age started smoking in former smokers'].replace("Prefer not to answer", pd.np.nan)
        smoking_data["Age started smoking in former smokers"] = pd.to_numeric(smoking_data["Age started smoking in former smokers"])

        smoking_data['Number of cigarettes currently smoked daily (current cigarette smokers)'] = smoking_data['Number of cigarettes currently smoked daily (current cigarette smokers)'].replace("Do not know", pd.np.nan)
        smoking_data['Number of cigarettes currently smoked daily (current cigarette smokers)'] = smoking_data['Number of cigarettes currently smoked daily (current cigarette smokers)'].replace("Prefer not to answer", pd.np.nan)
        smoking_data['Number of cigarettes currently smoked daily (current cigarette smokers)'] = smoking_data['Number of cigarettes currently smoked daily (current cigarette smokers)'].replace("Less than one a day", pd.np.nan)
        smoking_data["Number of cigarettes currently smoked daily (current cigarette smokers)"] = pd.to_numeric(smoking_data["Number of cigarettes currently smoked daily (current cigarette smokers)"])

        smoking_data['Number of cigarettes previously smoked daily'] = smoking_data['Number of cigarettes previously smoked daily'].replace("Do not know", pd.np.nan)
        smoking_data['Number of cigarettes previously smoked daily'] = smoking_data['Number of cigarettes previously smoked daily'].replace("Prefer not to answer", pd.np.nan)
        smoking_data['Number of cigarettes previously smoked daily'] = smoking_data['Number of cigarettes previously smoked daily'].replace("Less than one a day", pd.np.nan)
        smoking_data["Number of cigarettes previously smoked daily"] = pd.to_numeric(smoking_data["Number of cigarettes previously smoked daily"])
        return smoking_data
    
    def pre_white_bloodcell(self):
        '''
        Creates dataframe with white bloodcell data

        Returns
        -------
        df : pandas dataframe object
        '''
        wb = pd.read_csv(self.white_bloodcell)
        first_and_general_instances = [i for i in list(wb.columns) if not re.search("Instance 1|Instance 2|Instance 3", i)]
        remove_columns = [ i for i in wb.columns if i not in first_and_general_instances ]
        wb = wb.drop(remove_columns, axis=1)
        return wb
   
    def pre_symptomes(self):
        '''
        Creates dataframe with symptom data

        Returns
        -------
        df : pandas dataframe object
        '''
        s = pd.read_csv(self.symptomes, low_memory=False)
        s = s[['Participant ID',
                'Wheeze or whistling in the chest in last year | Instance 0'
                ]]
        s = s.replace("Prefer not to answer", pd.np.nan)
        s = s.replace("Do not know", pd.np.nan)
        
        #s = s.dropna()
        s = pd.get_dummies(s, columns=['Wheeze or whistling in the chest in last year | Instance 0'])
        return s
    
    def pre_attendance(self):
        '''
        Creates dataframe with attendance data

        Returns
        -------
        df : pandas dataframe object
        '''
        return pd.read_csv(self.attendance)
    
    def labeling_diabetes(self):
        '''
        Returns dataframe with two labels:
            1) first_occurence_binary: label which indicates whether a first occurence of diabetes has happend (no=0, yes=1)
            2) first_occurence_tertiary: label which indicates whether a first occurence has happend, is going to happen,
                                         or doesn't happen (0: did not happen, 1: did happen, 3: is going to happen)
        
        Returns
        -------
        df : pandas dataframe object
        ''' 
        df = pd.read_csv(self.first_occurence_diabetes)
        df["first_occurence_diabetes"] = df[df.columns[1:]].apply(self.earliest_date, axis=1) #Perform the earliest_date function
        df["first_occurence_diabetes_binary"] = df["first_occurence_diabetes"].apply(self.nan_to_binary) #Perform the nan_to_binary function

        att = pd.read_csv(self.attendance)
        att = att[["Participant ID", "Date of attending assessment centre | Instance 0"]] #Select columns
        df = df.merge(att, on = "Participant ID")
        df["first_occurence_diabetes_tertiary"] = df[["first_occurence_diabetes", "Date of attending assessment centre | Instance 0"]].apply(self.nan_to_tertiary, axis=1) #Apply nan_to_tertiary function

        di = ['Date E10 first reported (insulin-dependent diabetes mellitus)',
       'Date E11 first reported (non-insulin-dependent diabetes mellitus)',
       'Date E12 first reported (malnutrition-related diabetes mellitus)',
       'Date E13 first reported (other specified diabetes mellitus)',
       'Date E14 first reported (unspecified diabetes mellitus)']
        for i in di:
            df[i+"_onehot"] = df[i].apply(self.nan_to_binary) #One hot encoding for various diabetes types


        df = df.drop(["Date of attending assessment centre | Instance 0"], axis=1)
        return df
    
    def earliest_date(self, l):
        '''
        Returns the earliest date when provided by multiple in a list

        Returns
        -------
        l : list
            List with datetime objects

        Returns
        -------
        datetime object
            Earliest date
        '''
        l = l.dropna()
        if len(l) == 0:
            return pd.np.nan
        elif len(l) == 1:
            l = l[0].split("-")
            try:
                l = datetime.datetime(int(l[0]), int(l[1]), int(l[2]))
                return l
            except:
                print("Varible not convertable to datetime: ", l)
        else:
            t = []
            for i in l:
                i = i.split("-")
                try: 
                    i = datetime.datetime(int(i[0]), int(i[1]), int(i[2]))
                    t.append(i)
                except:
                    print("Varible not convertable to datetime: ", l)
            return min(t)
        
    def nan_to_binary(self,i):
        '''
        Convert a variable into binary on the existance of none object

        Parameters
        -------
        i : any variable

        Returns
        -------
        integer
            Binary encoding for nan
        '''
        if pd.isnull(i):
            return 0
        return 1
    
    def nan_to_tertiary(self,l):
        if pd.isnull(l[0]):
            return 0
        if pd.Timestamp(l[0]) < pd.Timestamp(l[1]):
            return 1
        return 2
    
    def labeling_asthma(self):
        '''
        Returns dataframe with class labels for asthma

        Returns
        -------
        df : pandas dataframe object
        '''
        #Obtain first occurence for asthma 
        df = pd.read_csv(self.first_occurence_asthma)
        
        #Obtain coding
        coding819 = pd.read_csv(self.coding819, sep="\t")
        self.coding = coding819["coding"].tolist()
        self.meaning = coding819["meaning"].tolist()

        #Delete with specific coding
        df= df[df["Date J45 first reported (asthma)"].apply(self.delete_coding) ==1]
        df= df[df["Date J46 first reported (status asthmaticus)"].apply(self.delete_coding) ==1]

        #Earliest date for first occurence
        df["all_asthma"] = df[["Date J45 first reported (asthma)", "Date J46 first reported (status asthmaticus)"]].apply(self.earliest_date, 1)
        df["all_asthma_binary"] = df["all_asthma"].apply(self.nan_to_binary)

        #Obtain assesment dates
        df = df.merge(pd.read_csv(self.attendance)[["Participant ID", "Date of attending assessment centre | Instance 0",
                                                     "Year of birth"]], on="Participant ID")
        df['Date of attending assessment centre | Instance 0']= pd.to_datetime(df['Date of attending assessment centre | Instance 0'])
        df["binary_assesment"] = df[["all_asthma", 'Date of attending assessment centre | Instance 0']].apply(self.binary_assesment, 1)

        #Age of asthma diagnosis
        df["age_asthma"] = df[["all_asthma", "Year of birth"]].apply(self.age_asthma, 1)

        df = df.drop(["Date of attending assessment centre | Instance 0"], axis=1)
        df = df.drop([ "Year of birth"], axis=1)

        return df
    
    def delete_coding(self, i):
        if i in self.meaning:
            return 0
        return 1
    
    def binary_assesment(self, l):
        if l[0] > l[1] and not pd.isnull(l[0]) and not pd.isnull(l[1]):
            return 1
        return 0
    
    def age_asthma(self, l):
        try:
            return l[0].year - l[1]
        except:
            return pd.np.nan
        
    def labeling_copd(self):
        '''
        Returns dataframe with class labels for copd

        Returns
        -------
        df : pandas dataframe object
        '''
        df = pd.read_csv(self.first_occurence_copd)

        cols = [
       'Date J40 first reported (bronchitis, not specified as acute or chronic)',
       'Date J41 first reported (simple and mucopurulent chronic bronchitis)',
       'Date J42 first reported (unspecified chronic bronchitis)',
       'Date J43 first reported (emphysema)',
       'Date J44 first reported (other chronic obstructive pulmonary disease)',
       'Date J47 first reported (bronchiectasis)']
        for i in cols:
            df[i+"_binary"] = df[i].apply(self.nan_to_binary)
        
        cols_binary = [i+"_binary" for i in cols]
        df["all_copd"] = df[cols_binary].apply(self.any_binary, 1)

        coding819 = pd.read_csv(self.coding819, sep="\t")
        self.coding = coding819["coding"].tolist()
        self.meaning = coding819["meaning"].tolist()
        for i in cols:
            df = df[df[i].apply(self.delete_coding) ==1]

        df["all_copd_date"] = df[cols].apply(self.earliest_date, 1)

        return df
    
    def any_binary(self, l):
        for i in l:
            if i == 1:
                return 1
        return 0
    
    def labeling_osteoporosis(self):
        '''
        Returns dataframe with class labels for osteoporosis

        Returns
        -------
        df : pandas dataframe object
        '''
        df = pd.read_csv(self.first_occurence_osteoporosis)
        att = pd.read_csv(self.attendance)
        df = df.merge(att[['Participant ID', 'Year of birth']], on='Participant ID')

        df["Date M80 first reported (osteoporosis with pathological fracture)_binary"] = df["Date M80 first reported (osteoporosis with pathological fracture)"].apply(self.nan_to_binary)
        df["Date M81 first reported (osteoporosis without pathological fracture)_binary"] = df["Date M81 first reported (osteoporosis without pathological fracture)"].apply(self.nan_to_binary)
        df["Date M82 first reported (osteoporosis in diseases classified elsewhere)_binary"] = df["Date M82 first reported (osteoporosis in diseases classified elsewhere)"].apply(self.nan_to_binary)
    
        df[ 'Date M80 first reported (osteoporosis with pathological fracture)_year'] = pd.DatetimeIndex(df['Date M80 first reported (osteoporosis with pathological fracture)']).year
        df[ 'Date M81 first reported (osteoporosis without pathological fracture)_year'] = pd.DatetimeIndex(df['Date M81 first reported (osteoporosis without pathological fracture)']).year
        df[ 'Date M82 first reported (osteoporosis in diseases classified elsewhere)_year'] = pd.DatetimeIndex(df['Date M82 first reported (osteoporosis in diseases classified elsewhere)']).year

        df["Date M80 first reported (osteoporosis with pathological fracture)_age"] = df[['Year of birth', 'Date M80 first reported (osteoporosis with pathological fracture)_year']].apply(self.age_osteo, axis=1)
        df["Date M81 first reported (osteoporosis without pathological fracture)_age"] = df[['Year of birth', 'Date M80 first reported (osteoporosis with pathological fracture)_year']].apply(self.age_osteo, axis=1)
        df["Date M82 first reported (osteoporosis in diseases classified elsewhere)_age"] = df[['Year of birth', 'Date M80 first reported (osteoporosis with pathological fracture)_year']].apply(self.age_osteo, axis=1)
        #TODO: Add all_osteo column
        df = df.drop(['Year of birth'], axis=1)
        return df
    
    def age_osteo(self, l):
        return l[1]-l[0]
    
    def labeling_cvd(self):
        '''
        Returns dataframe with class labels for cvd

        Returns
        -------
        df : pandas dataframe object
        '''
        cvd = pd.read_csv(self.first_occurence_cvd, low_memory=False)
        cvd_columns = cvd.columns.tolist()
        cvd_columns.remove("Participant ID")
        att = pd.read_csv(self.attendance)
        att['Date of attending assessment centre | Instance 0'] = pd.to_datetime(att['Date of attending assessment centre | Instance 0'])
        cvd = cvd.merge(att[["Participant ID", 'Date of attending assessment centre | Instance 0']], on="Participant ID")

        for c in cvd_columns:
            cvd[c+"_binary"] = cvd[c].apply(self.nan_to_binary)

        for c in cvd_columns:
            cvd[c] = cvd[c].replace("Code has event date matching participant's date of birth", pd.np.nan)
            cvd[c] = cvd[c].replace("Code has event date after participant's date of birth and falls in the same calendar year as date of birth", pd.np.nan)
            cvd[c] = pd.to_datetime(cvd[c])
            cvd[c+'diff_days'] = (cvd[c] - cvd['Date of attending assessment centre | Instance 0']) / np.timedelta64(1, 'D')
        cvd = cvd.drop(['Date of attending assessment centre | Instance 0'], axis=1)
        return cvd
    
    def factory(self):
        #Data
        print("DATAPREPROCESSING INITIALIZED")
        print("0/19")
        df = self.pre_demographics_basic()
        print("1/19")
        df = df.merge(self.pre_demographics_ethnicity() ,on="Participant ID")
        print("2/19")
        df = self.pre_blood_biomarker(dem_basic=df) 
        print("3/19")
        df = df.merge(self.pre_alcohol() ,on="Participant ID") 
        print("4/19")
        df = df.merge(self.pre_bodymeasures() ,on="Participant ID")
        print("5/19")
        df = df.merge(self.pre_blood_pressure() ,on="Participant ID")
        print("6/19")
        df = df.merge(self.pre_family_history() ,on="Participant ID")
        print("7/19")
        df = df.merge(self.pre_medical_conditions() ,on="Participant ID")
        print("9/19")
        df = df.merge(self.pre_sleep() ,on="Participant ID")
        print("10/19")
        df = df.merge(self.pre_smoking() ,on="Participant ID")
        print("11/19")
        df = df.merge(self.pre_urine_biomarkers() ,on="Participant ID")
        print("12/19")
        df = df.merge(self.pre_physical_activity() ,on="Participant ID")
        print("13/19")
        df = df.merge(self.pre_white_bloodcell() ,on="Participant ID")
        print("14/19")
        df = df.merge(self.pre_smoking_supplementary() ,on="Participant ID")
        print("15/19")
        df = df.merge(self.pre_symptomes() ,on="Participant ID")
        df = df.merge(self.pre_attendance(),on="Participant ID")

        #Labels
        df = df.merge(self.labeling_diabetes() ,on="Participant ID")
        print("16/19")
        df = df.merge(self.labeling_copd() ,on="Participant ID")
        print("17/19")
        df = df.merge(self.labeling_asthma() ,on="Participant ID")
        print("18/19")
        df = df.merge(self.labeling_osteoporosis() ,on="Participant ID")
        df = df.merge(self.labeling_cvd(), on="Participant ID")
        print("19/19 \nDATAPREPROCESSING DONE")
        
        self.df = df
    


class ClusterWrapper():
    '''
    Wrapper for clustering model
    '''
    def __init__(self, cluster_model, scaler):
        self.cluster_model = cluster_model
        self.scaler = scaler


class RFWrapper():
    '''
    Wrapper for random forest model
    '''
    def __init__(self, rf_model, scaler, min_days, max_days) :#TODO: Add name property
        self.rf_model = rf_model
        self.scaler = scaler
        self.min_days = min_days
        self.max_days = max_days


class NBWrapper():
    '''
    Wrapper for Naive bayes model
    '''
    def __init__(self, nb_model): 
        self.nb_model = nb_model


class LRWrapper():
    '''
    Wrapper for logistic regression model
    '''
    def __init__(self, lr_model, scaler, label_column, data_columns): 
        self.lr_model = lr_model
        self.scaler = scaler
        self.label_column = label_column
        self.data_columns = data_columns
     
        
class modelConstruction():

    def __init__(self, df, test, train, evaluation_folder, dp, sex="", boxplot_eval=False, results=pd.DataFrame()):
        '''
        Is dependent on dataPreprocessing
        '''
        #Obtain data
        self.df = df
        self.test = test
        self.train = train
        self.train = self.train.reset_index() 
        self.evaluation_folder = evaluation_folder
        self.dp = dp 
        self.sex = sex
        self.boxplot_eval = boxplot_eval
        self.results = results

        #Diabetes models
        self.niddm_na_one_lada_clustermodel = ClusterWrapper(None, None)
        self.niddm_na_five_lada_clustermodel = ClusterWrapper(None, None)
        self.niddm_na_fiveten_lada_clustermodel = ClusterWrapper(None, None)
        self.niddm_na_ten_lada_clustermodel = ClusterWrapper(None, None)
        self.niddm_na_one_lada_rfmodel_one = RFWrapper(None, None, 0, 365)
        self.niddm_na_one_lada_rfmodel_two = RFWrapper(None, None, 0, 365)
        self.niddm_na_one_lada_rfmodel_three = RFWrapper(None, None, 0, 365)
        self.niddm_na_one_lada_rfmodel_four = RFWrapper(None, None, 0, 365)
        self.niddm_na_five_lada_rfmodel_one = RFWrapper(None, None, 0, 1825)
        self.niddm_na_five_lada_rfmodel_two = RFWrapper(None, None, 0, 1825)
        self.niddm_na_five_lada_rfmodel_three = RFWrapper(None, None, 0, 1825)
        self.niddm_na_five_lada_rfmodel_four = RFWrapper(None, None, 0, 1825)
        self.niddm_na_fiveten_lada_rfmodel_one = RFWrapper(None, None, 1825, 3650)
        self.niddm_na_fiveten_lada_rfmodel_two = RFWrapper(None, None, 1825, 3650)
        self.niddm_na_fiveten_lada_rfmodel_three = RFWrapper(None, None, 1825, 3650)
        self.niddm_na_fiveten_lada_rfmodel_four = RFWrapper(None, None, 1825, 3650)
        self.niddm_na_ten_lada_rfrmodel_one = RFWrapper(None, None, 3650, 99999999)
        self.niddm_na_ten_lada_rfrmodel_two = RFWrapper(None, None, 3650, 99999999)
        self.niddm_na_ten_lada_rfrmodel_three = RFWrapper(None, None, 3650, 99999999)
        self.niddm_na_ten_lada_rfrmodel_four = RFWrapper(None, None, 3650, 99999999)
        
        self.diabetes_data_columns = None
        self.diabetes_question_columns = None

        #COPD models
        self.emphysema_model_current = LRWrapper(None, None, None, None)
        self.chronic_bronchitis_model_current = LRWrapper(None, None, None, None)
        self.other_copd_model_current = LRWrapper(None, None, None, None)

        self.emphysema_model_past = LRWrapper(None, None, None, None)
        self.other_copd_model_past = LRWrapper(None, None, None, None)

        self.copd_data_columns = None
        self.copd_cluster_columns = None

        #Osteoporosis
        #Linear regression:
        self.slope_osteo = None 
        self.intercept_osteo = None
        self.binary_columns_osteo = None

        self.nb_osteo = NBWrapper(None)

        #Asthma
        self.binary_columns_asthma = None
        self.slope_asthma = None
        self.intercept_asthma = None

        self.nb_atshma = NBWrapper(None)

        #CVD
        self.cvd_hf_model = LRWrapper(None, None, None, None)
        self.cvd_isch_model = LRWrapper(None, None, None, None)

    def diabetes_modeling(self):
        '''
        Diabetes model construction
        '''
        #Dataprep
        niddm = self.train[(self.train['Date E11 first reported (non-insulin-dependent diabetes mellitus)_onehot'] == 1)]
        healthy = self.train[(self.train['first_occurence_diabetes_binary'] == 0) & (self.train['Glycated haemoglobin (HbA1c) | Instance 0'] < 48)]
        question = [
                'Age at recruitment',
                'Sex',
                'Asian',
                'Black',
                'Chinese',
                'Mixed',
                'Other',
                'White',
                'Alcohol intake frequency. | Instance 0_Daily or almost daily',
                'Alcohol intake frequency. | Instance 0_Never',
                'Alcohol intake frequency. | Instance 0_Once or twice a week',
                'Alcohol intake frequency. | Instance 0_One to three times a month',
                'Alcohol intake frequency. | Instance 0_Special occasions only',
                'Alcohol intake frequency. | Instance 0_Three or four times a week',
                'fmi',
                'Body mass index (BMI) | Instance 0',
                'Body fat percentage | Instance 0',
                'Waist circumference | Instance 0',
                'Weight | Instance 0',
                'Hip circumference | Instance 0',
                'Whole body fat mass | Instance 0',
                'Basal metabolic rate | Instance 0',
                'Trunk fat percentage | Instance 0',
                'Arm fat percentage (left) | Instance 0',
                'Leg fat percentage (left) | Instance 0',
                'Diastolic blood pressure',
                'Systolic blood pressure',
                'Illnesses of father',
                'Illnesses of mother',
                'Illnesses of siblings',
                'Cholesterol_lowering_medication',
                'Insulin',
                'Blood_pressure',
                'Sleep duration | Instance 0',
                'Tobacco smoking_Ex-smoker',
                'Tobacco smoking_Never smoked',
                'Tobacco smoking_Occasionally',
                'Tobacco smoking_Smokes on most or all days',
                'Summed MET minutes per week for all activity | Instance 0',
                'Summed minutes activity | Instance 0'
                ]
        
        self.diabetes_question_columns = question
        cols_both = question+['Glycated haemoglobin (HbA1c) | Instance 0']
        cols = question+['Glycated haemoglobin (HbA1c) | Instance 0','first_occurence_diabetes_binary', 
                         'Date E11 first reported (non-insulin-dependent diabetes mellitus)',
                         'Date of attending assessment centre | Instance 0']
        
        niddm_na = niddm[cols].dropna()
        healthy_na = healthy[cols_both].dropna()
        niddm_na = niddm_na[(niddm_na['Date E11 first reported (non-insulin-dependent diabetes mellitus)'] != "Code has event date matching participant's date of birth")]

        niddm_na['Date of attending assessment centre | Instance 0'] = pd.to_datetime(niddm_na['Date of attending assessment centre | Instance 0'])
        niddm_na['Date E11 first reported (non-insulin-dependent diabetes mellitus)'] = pd.to_datetime(niddm_na['Date E11 first reported (non-insulin-dependent diabetes mellitus)'])

        niddm_na['diff_days'] = (niddm_na['Date E11 first reported (non-insulin-dependent diabetes mellitus)'] - niddm_na['Date of attending assessment centre | Instance 0']) / np.timedelta64(1, 'D')

        #Different timescales
        niddm_na_five = niddm_na[(niddm_na['diff_days'] < 1825) & (niddm_na['diff_days'] > 0)]
        niddm_na_one = niddm_na[(niddm_na['diff_days'] < 365) & (niddm_na['diff_days'] > 0)]
        niddm_na_fiveten = niddm_na[(niddm_na['diff_days'] > 1825) & (niddm_na['diff_days'] < 3650)]
        niddm_na_ten = niddm_na[(niddm_na['diff_days'] > 3650)]

        question_hb1ac = question+['Glycated haemoglobin (HbA1c) | Instance 0'] 

        #LADA column creation
        niddm_na_one["LADA"] = niddm_na_one[['Age at recruitment','Body mass index (BMI) | Instance 0', 'Glycated haemoglobin (HbA1c) | Instance 0']].apply(self.lada_alg,axis=1)
        niddm_na_five["LADA"] = niddm_na_five[['Age at recruitment','Body mass index (BMI) | Instance 0', 'Glycated haemoglobin (HbA1c) | Instance 0']].apply(self.lada_alg,axis=1)
        niddm_na_fiveten["LADA"] = niddm_na_fiveten[['Age at recruitment','Body mass index (BMI) | Instance 0', 'Glycated haemoglobin (HbA1c) | Instance 0']].apply(self.lada_alg,axis=1)
        niddm_na_ten["LADA"] = niddm_na_ten[['Age at recruitment','Body mass index (BMI) | Instance 0', 'Glycated haemoglobin (HbA1c) | Instance 0']].apply(self.lada_alg,axis=1)

        #Exclusion on LADA
        niddm_na_one_lada = niddm_na_one[(niddm_na_one["LADA"] == 0)]
        niddm_na_five_lada = niddm_na_five[(niddm_na_five["LADA"] == 0)]
        niddm_na_fiveten_lada = niddm_na_fiveten[(niddm_na_fiveten["LADA"] == 0)]
        niddm_na_ten_lada = niddm_na_ten[(niddm_na_ten["LADA"] == 0)]

        #Cluster model construction
        self.niddm_na_one_lada_clustermodel.cluster_model, niddm_na_one_lada_clustercolumn, self.niddm_na_one_lada_clustermodel.scaler = self.kmeans_clustering(niddm_na_one_lada,question, 4)
        self.niddm_na_five_lada_clustermodel.cluster_model, niddm_na_five_lada_clustercolumn, self.niddm_na_five_lada_clustermodel.scaler = self.kmeans_clustering(niddm_na_five_lada,question, 4)
        self.niddm_na_fiveten_lada_clustermodel.cluster_model, niddm_na_fiveten_lada_clustercolumn, self.niddm_na_fiveten_lada_clustermodel.scaler = self.kmeans_clustering(niddm_na_fiveten_lada,question, 4)
        self.niddm_na_ten_lada_clustermodel.cluster_model, niddm_na_ten_lada_clustercolumn, self.niddm_na_ten_lada_clustermodel.scaler = self.kmeans_clustering(niddm_na_ten_lada,question, 4)

        #Adding clustering results
        niddm_na_one_lada['cluster'] = niddm_na_one_lada_clustercolumn
        niddm_na_five_lada['cluster'] = niddm_na_five_lada_clustercolumn
        niddm_na_fiveten_lada['cluster'] = niddm_na_fiveten_lada_clustercolumn
        niddm_na_ten_lada['cluster'] = niddm_na_ten_lada_clustercolumn

        model_cols = cols_both+["cluster"]
        self.diabetes_data_columns = cols_both

        #Model construction
        x = self.cluster_label(self.class_balance(niddm_na_one_lada[niddm_na_one_lada['cluster']==0][model_cols], healthy_na))
        self.niddm_na_one_lada_rfmodel_one.rf_model, self.niddm_na_one_lada_rfmodel_one.scaler = self.random_forest(y_train=x['cluster_label'], x_train=x[cols_both])

        x = self.cluster_label(self.class_balance(niddm_na_one_lada[niddm_na_one_lada['cluster']==1][model_cols], healthy_na))
        self.niddm_na_one_lada_rfmodel_two.rf_model, self.niddm_na_one_lada_rfmodel_two.scaler = self.random_forest(y_train=x['cluster_label'], x_train=x[cols_both])

        x = self.cluster_label(self.class_balance(niddm_na_one_lada[niddm_na_one_lada['cluster']==2][model_cols], healthy_na))
        self.niddm_na_one_lada_rfmodel_three.rf_model, self.niddm_na_one_lada_rfmodel_three.scaler = self.random_forest(y_train=x['cluster_label'], x_train=x[cols_both])

        x = self.cluster_label(self.class_balance(niddm_na_one_lada[niddm_na_one_lada['cluster']==3][model_cols], healthy_na))
        self.niddm_na_one_lada_rfmodel_four.rf_model, self.niddm_na_one_lada_rfmodel_four.scaler = self.random_forest(y_train=x['cluster_label'], x_train=x[cols_both])

        x = self.cluster_label(self.class_balance(niddm_na_five_lada[niddm_na_five_lada['cluster']==0][model_cols], healthy_na))
        self.niddm_na_five_lada_rfmodel_one.rf_model, self.niddm_na_five_lada_rfmodel_one.scaler = self.random_forest(y_train=x['cluster_label'], x_train=x[cols_both])

        x = self.cluster_label(self.class_balance(niddm_na_five_lada[niddm_na_five_lada['cluster']==1][model_cols], healthy_na))
        self.niddm_na_five_lada_rfmodel_two.rf_model, self.niddm_na_five_lada_rfmodel_two.scaler = self.random_forest(y_train=x['cluster_label'], x_train=x[cols_both])

        x= self.cluster_label(self.class_balance(niddm_na_five_lada[niddm_na_five_lada['cluster']==2][model_cols], healthy_na))
        self.niddm_na_five_lada_rfmodel_three.rf_model, self.niddm_na_five_lada_rfmodel_three.scaler = self.random_forest(y_train=x['cluster_label'], x_train=x[cols_both])

        x = self.cluster_label(self.class_balance(niddm_na_five_lada[niddm_na_five_lada['cluster']==3][model_cols], healthy_na))
        self.niddm_na_five_lada_rfmodel_four.rf_model, self.niddm_na_five_lada_rfmodel_four.scaler = self.random_forest(y_train=x['cluster_label'], x_train=x[cols_both])

        x = self.cluster_label(self.class_balance(niddm_na_fiveten_lada[niddm_na_fiveten_lada['cluster']==0][model_cols], healthy_na))
        self.niddm_na_fiveten_lada_rfmodel_one.rf_model, self.niddm_na_fiveten_lada_rfmodel_one.scaler = self.random_forest(y_train=x['cluster_label'], x_train=x[cols_both])

        x = self.cluster_label(self.class_balance(niddm_na_fiveten_lada[niddm_na_fiveten_lada['cluster']==1][model_cols], healthy_na))
        self.niddm_na_fiveten_lada_rfmodel_two.rf_model, self.niddm_na_fiveten_lada_rfmodel_two.scaler = self.random_forest(y_train=x['cluster_label'], x_train=x[cols_both])

        x = self.cluster_label(self.class_balance(niddm_na_fiveten_lada[niddm_na_fiveten_lada['cluster']==2][model_cols], healthy_na))
        self.niddm_na_fiveten_lada_rfmodel_three.rf_model, self.niddm_na_fiveten_lada_rfmodel_three.scaler = self.random_forest(y_train=x['cluster_label'], x_train=x[cols_both])

        x = self.cluster_label(self.class_balance(niddm_na_fiveten_lada[niddm_na_fiveten_lada['cluster']==3][model_cols], healthy_na))
        self.niddm_na_fiveten_lada_rfmodel_four.rf_model, self.niddm_na_fiveten_lada_rfmodel_four.scaler = self.random_forest(y_train=x['cluster_label'], x_train=x[cols_both])

        x = self.cluster_label(self.class_balance(niddm_na_ten_lada[niddm_na_ten_lada['cluster']==0][model_cols], healthy_na))
        self.niddm_na_ten_lada_rfrmodel_one.rf_model, self.niddm_na_ten_lada_rfrmodel_one.scaler = self.random_forest(y_train=x['cluster_label'], x_train=x[cols_both])

        x = self.cluster_label(self.class_balance(niddm_na_ten_lada[niddm_na_ten_lada['cluster']==1][model_cols], healthy_na))
        self.niddm_na_ten_lada_rfrmodel_two.rf_model, self.niddm_na_ten_lada_rfrmodel_two .scaler = self.random_forest(y_train=x['cluster_label'], x_train=x[cols_both])

        x = self.cluster_label(self.class_balance(niddm_na_ten_lada[niddm_na_ten_lada['cluster']==2][model_cols], healthy_na))
        self.niddm_na_ten_lada_rfrmodel_three.rf_model, self.niddm_na_ten_lada_rfrmodel_three.scaler = self.random_forest(y_train=x['cluster_label'], x_train=x[cols_both])

        x = self.cluster_label(self.class_balance(niddm_na_ten_lada[niddm_na_ten_lada['cluster']==3][model_cols], healthy_na))
        self.niddm_na_ten_lada_rfrmodel_four.rf_model, self.niddm_na_ten_lada_rfrmodel_four.scaler = self.random_forest(y_train=x['cluster_label'], x_train=x[cols_both])
        
        #Creating boxplots
        if self.boxplot_eval:
            for c in question_hb1ac: 
                self.cluster_boxplot(title=c+' niddm_na_one_lada_'+self.sex, data=[niddm_na_one_lada[niddm_na_one_lada['cluster']==n][c].tolist() for n in range(0,4)])
                self.cluster_boxplot(title=c+' niddm_na_five_lada_'+self.sex, data=[niddm_na_five_lada[niddm_na_five_lada['cluster']==n][c].tolist() for n in range(0,4)])
                self.cluster_boxplot(title=c+' niddm_na_fiveten_lada_'+self.sex, data=[niddm_na_fiveten_lada[niddm_na_fiveten_lada['cluster']==n][c].tolist() for n in range(0,4)])
                self.cluster_boxplot(title=c+' niddm_na_ten_lada_'+self.sex, data=[niddm_na_ten_lada[niddm_na_ten_lada['cluster']==n][c].tolist() for n in range(0,4)])


    def lada_alg(self, l):
        '''
        Excludes individuals based on LADA criteria

        Parameters
        -------
        l : list
            List with bmi values

        Returns
        -------
        integer
            Binary encoding for LADA
        ''' 
        if l[1] < 24:
                return 1
        return 0
    
    def kmeans_clustering(self,df ,data_columns, n_clusters, scaling=True, random_state=0 ):
        '''
        Constructs kmeans clustering model

        Parameters
        -------
        df : pandas dataframe object
            Datasource
        n_clusters : int
            Number of clusters to be generated
        scaling : Binary
            Indicates wether or not data should be scaled (default is True)
        random_state : 0
            Random state of kmeans algorithm (default is 0)

        Returns
        -------
        kmeans : sklearn kmeans model
        kmeans label : list
            Labels for all data as generated by kmeans model
        mms : minmaxscaler object
        '''
        tdf = df[data_columns].dropna()
        mms = None
        if scaling:
            mms = MinMaxScaler()
            mms.fit(tdf[data_columns])
            data_transformed = mms.transform(tdf[data_columns])
            tdf = pd.DataFrame(data_transformed, columns=data_columns)

        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto").fit(tdf)
        return kmeans, kmeans.labels_.tolist(), mms
    
    def class_balance(self, targetclass, class2, ratio_multiplyer=1):
        '''
        Balances classes based on different dataframes

        Parameters
        -------
        targetclass : pandas dataframe object
            Dataframe object with the target class
        class2 : pandas dataframe object
            Dataframe containing the other class
        ratio_mulitplyer : int
            Defines desired ratio between classes (default is 1)

        Returns
        -------
        cohort : pandas dataframe object
        '''
        if targetclass.shape[0] > class2.shape[0]:
            cohort = pd.concat([class2, targetclass.sample(n=round(class2.shape[0]*ratio_multiplyer))], axis=0)
        else:
            cohort = pd.concat([targetclass, class2.sample(n=round(targetclass.shape[0]*ratio_multiplyer))], axis=0)
        return cohort
    
    def cluster_label(self, cohort, clustertag='cluster'):
        cohort[clustertag+'_label'] = cohort[clustertag].apply(self.dp.nan_to_binary)
        return cohort

    def random_forest(self, x_train, y_train, scaling=True):
        '''
        Constructs random forest model

        Parameters
        -------
        x_train : pandas dataframe object
            Datasource
        y_train : pandas dataframe column
            Column containing class variable
        scaling : Binary
            Indicates wether or not data should be scaled (default is True)
    
        Returns
        -------
        model : sklearn random forest model
        mms : minmaxscaler object
        '''
        mms = None
        if scaling:
            mms = MinMaxScaler()
            mms.fit(x_train)
            tcol = x_train.columns.tolist()
            data_transformed = mms.transform(x_train)
            x_train = pd.DataFrame(data_transformed, columns=tcol)
        model = RandomForestClassifier()
        model.fit(x_train, y_train)
        return model, mms
    
    def logistic_regression(self, x_train, y_train, scaling=True):
        '''
        Constructs logistic regression model

        Parameters
        -------
        x_train : pandas dataframe object
            Datasource
        y_train : pandas dataframe column
            Column containing class variable
        scaling : Binary
            Indicates wether or not data should be scaled (default is True)
    
        Returns
        -------
        model : sklearn logistic regression model
        mms : minmaxscaler object
        '''
        mms = None
        if scaling:
            mms = MinMaxScaler()
            mms.fit(x_train)
            tcol = x_train.columns.tolist()
            data_transformed = mms.transform(x_train)
            x_train = pd.DataFrame(data_transformed, columns=tcol)
        logisticRegr = LogisticRegression(max_iter=1000000000)
        logisticRegr.fit(x_train, y_train)
        return logisticRegr, mms

    def cluster_boxplot(self, title, data, ylabel=""):
        '''
        Creater boxplots with are saved to standard folder

        Parameters
        -------
        title : string
            Title of the plot
        data : List
            List with nested lists which contain plotting data
        '''
        fig = plt.figure()
        
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_title(title)
        bp = ax.boxplot(data)
        for i in ["/", "/", "\\", "<", ">", ":","|", "?"]:
            title = title.replace(i,"")
        fig.savefig(self.evaluation_folder+title+'.png', bbox_inches='tight')
        plt.close()

   
    def copd_modeling(self):
        '''
        This fucntion creates COPD models
        '''
        #Define various columns for current smokers
        self.chronic_bronchitis_model_current.data_columns = [
            'Age at recruitment',
            'Pack years adult smoking as proportion of life span exposed to smoking',
            'Brisk pace',
            'Steady average pace',
            'Number of cigarettes previously smoked daily',
            'hayfever/allergic rhinitis',
            'Age started smoking in former smokers',
            'allergy or anaphylactic reaction to food',
            'Pack years of smoking',
            'Neutrophill count | Instance 0'
        ]
        self.chronic_bronchitis_model_current.label_column = 'Date J42 first reported (unspecified chronic bronchitis)_binary'

        self.emphysema_model_current.data_columns = [
            'Pack years adult smoking as proportion of life span exposed to smoking',
            'Age at recruitment',
            'Number of cigarettes previously smoked daily',
            'Brisk pace',
            'Body mass index (BMI) | Instance 0',
            'chronic sinusitis',
            'Steady average pace',
            'Neutrophill count | Instance 0',
            'Pack years of smoking',
            'Hayfever_allergic_rhinitis_eczema_doctor'
        ]
        self.emphysema_model_current.label_column = 'Date J43 first reported (emphysema)_binary'
        
        self.other_copd_model_current.data_columns = [
            'Age at recruitment',
            'Brisk pace',
            'Pack years adult smoking as proportion of life span exposed to smoking',
            'Steady average pace',
            'Number of cigarettes previously smoked daily',
            'Pack years of smoking',
            'Neutrophill count | Instance 0',
            'Doctor diagnosed asthma',
            'Eosinophill count | Instance 0',
            'Summed MET minutes per week for all activity | Instance 0'
        ]
        self.other_copd_model_current.label_column = 'Date J44 first reported (other chronic obstructive pulmonary disease)_binary'

        #Model construction
        na = self.train[self.chronic_bronchitis_model_current.data_columns+[self.chronic_bronchitis_model_current.label_column]].dropna()
        na = self.class_balance(targetclass=na[na[self.chronic_bronchitis_model_current.label_column]==1], class2=na[na[self.chronic_bronchitis_model_current.label_column]==0])
        self.chronic_bronchitis_model_current.lr_model , self.chronic_bronchitis_model_current.scaler = self.logistic_regression(y_train=na[self.chronic_bronchitis_model_current.label_column],
                                                                                        x_train=na[self.chronic_bronchitis_model_current.data_columns],
                                                                                        scaling=False)
        
        na = self.train[self.emphysema_model_current.data_columns+[self.emphysema_model_current.label_column]].dropna()
        na = self.class_balance(targetclass=na[na[self.emphysema_model_current.label_column]==1], class2=na[na[self.emphysema_model_current.label_column]==0])
        self.emphysema_model_current.lr_model , self.emphysema_model_current.scaler = self.logistic_regression(y_train=na[self.emphysema_model_current.label_column],
                                                                                        x_train=na[self.emphysema_model_current.data_columns],
                                                                                        scaling=False)

        na = self.train[self.other_copd_model_current.data_columns+[self.other_copd_model_current.label_column ]].dropna()
        na = self.class_balance(targetclass=na[na[self.other_copd_model_current.label_column]==1], class2=na[na[self.other_copd_model_current.label_column]==0])
        self.other_copd_model_current.lr_model , self.other_copd_model_current.scaler = self.logistic_regression(y_train=na[self.other_copd_model_current.label_column ],
                                                                                        x_train=na[self.other_copd_model_current.data_columns],
                                                                                        scaling=False)

        #With data on past smokers
        self.emphysema_model_past.data_columns = [
            'Number of cigarettes currently smoked daily (current cigarette smokers)',
            'Age at recruitment',
            'Pack years of smoking',
            'Age started smoking in current smokers',
            'Body mass index (BMI) | Instance 0',
            'Brisk pace',
            'Steady average pace',
            'Pack years adult smoking as proportion of life span exposed to smoking',
            'Cholesterol | Instance 0',
            'Haemoglobin concentration | Instance 0'
        ]
        self.emphysema_model_past.label_column = 'Date J43 first reported (emphysema)_binary'

        self.other_copd_model_past.data_columns = [
            'Number of cigarettes currently smoked daily (current cigarette smokers)',
            'Age at recruitment',
            'Brisk pace',
            'Pack years adult smoking as proportion of life span exposed to smoking',
            'Age started smoking in current smokers',
            'Steady average pace',
            'Body mass index (BMI) | Instance 0',
            'Cholesterol | Instance 0',
            'asbestosis',
            'Haemoglobin concentration | Instance 0'
        ]
        self.other_copd_model_past.label_column = 'Date J44 first reported (other chronic obstructive pulmonary disease)_binary'

        na = self.train[self.emphysema_model_past.data_columns+[self.emphysema_model_past.label_column]].dropna()
        na = self.class_balance(targetclass=na[na[self.emphysema_model_past.label_column]==1], class2=na[na[self.emphysema_model_past.label_column]==0])
        self.emphysema_model_past.lr_model , self.emphysema_model_past.scaler = self.logistic_regression(y_train=na[self.emphysema_model_past.label_column],
                                                                                        x_train=na[self.emphysema_model_past.data_columns],
                                                                                        scaling=False)

        
        na = self.train[self.other_copd_model_past.data_columns+[self.other_copd_model_past.label_column]].dropna()
        na = self.class_balance(targetclass=na[na[self.other_copd_model_past.label_column]==1], class2=na[na[self.other_copd_model_past.label_column]==0])
        self.other_copd_model_past.lr_model , self.other_copd_model_past.scaler = self.logistic_regression(y_train=na[self.other_copd_model_past.label_column],
                                                                                        x_train=na[self.other_copd_model_past.data_columns],
                                                                                        scaling=False)

    def cvd_modeling(self):
        '''
        Model creation for CVD
        '''
        #Define used columns
        ten_hf = ['Hip circumference | Instance 0',
         'heart attack/myocardial infarction',
         'bronchitis',
         'nasal/sinus disorder',
         'fmi',
         'osteoarthritis',
         'Age at recruitment',
         'Body mass index (BMI) | Instance 0',
         'fracture upper arm / humerus / elbow',
         'rheumatic fever']
        self.cvd_hf_model.data_columns = ten_hf
        self.cvd_hf_model.label_column = 'Date I50 first reported (heart failure)_binary'
        
        ten_isch = [
        "heart attack/myocardial infarction",
        "angina",
        "Age at recruitment",
        "Basal metabolic rate | Instance 0",
        "ovarian cyst or cysts",
        "Pack years adult smoking as proportion of life span exposed to smoking",
        "essential hypertension",
        "Pack years of smoking",
        "Slow pace",
        "hepatitis c"]
        self.cvd_isch_model.data_columns = ten_isch
        self.cvd_isch_model.label_column = "Date I25 first reported (chronic ischaemic heart disease)_binary"

        #Model construction
        na = self.train[self.cvd_hf_model.data_columns+[self.cvd_hf_model.label_column, 'Date I50 first reported (heart failure)diff_days']].dropna(subset=self.cvd_hf_model.data_columns)
        na = self.class_balance(targetclass=na[  (na[self.cvd_hf_model.label_column]==1)
                                               & (na['Date I50 first reported (heart failure)diff_days'] > 0) 
                                               & (na['Date I50 first reported (heart failure)diff_days'] < 5*365)],
                                class2=na[na[self.cvd_hf_model.label_column]==0])
        self.cvd_hf_model.lr_model , self.cvd_hf_model.scaler = self.logistic_regression(y_train=na[self.cvd_hf_model.label_column],
                                                                                        x_train=na[self.cvd_hf_model.data_columns],
                                                                                        scaling=False)

        na = self.train[self.cvd_isch_model.data_columns+[self.cvd_isch_model.label_column, 'Date I25 first reported (chronic ischaemic heart disease)diff_days']].dropna(subset=self.cvd_isch_model.data_columns)
        na = self.class_balance(targetclass=na[  (na[self.cvd_isch_model.label_column]==1)
                                               & (na['Date I25 first reported (chronic ischaemic heart disease)diff_days'] > 0) 
                                               & (na['Date I25 first reported (chronic ischaemic heart disease)diff_days'] < 5*365)],
                                class2=na[na[self.cvd_isch_model.label_column]==0])
        self.cvd_isch_model.lr_model , self.cvd_isch_model.scaler = self.logistic_regression(y_train=na[self.cvd_isch_model.label_column],
                                                                                        x_train=na[self.cvd_isch_model.data_columns],
                                                                                        scaling=False)

    def lr_evaluation_cvd(self, model_name, lr_model,stratisfy=True):
        '''
        This function evaluates CVD LR models, results are written to evaluations file

        Parameters
        -------
        model_name : String
            Name of model
        lr_model : LR wrapper object
        Stratisfy : Binary
            Indicates if data should be class balanced (default is True)
        '''
        #Obtain testing data
        test = self.test[lr_model.data_columns+["Participant ID", lr_model.label_column, model_name+"diff_days"]].dropna()

        #Scale testing data
        if lr_model.scaler != None:
            test_id = test["Participant ID"].tolist()
            lc = test[lr_model.label_column].tolist()

            data_transformed = lr_model.scaler.transform(test[lr_model.data_columns])
            test= pd.DataFrame(data_transformed, columns=lr_model.data_columns)

            test['Participant ID'] = test_id
            test[lr_model.label_column] = lc

        #Stratification
        if stratisfy:
            test = pd.concat([test[test[lr_model.label_column]==1], test[test[lr_model.label_column]==0].sample(
                n=round(test[test[lr_model.label_column]==1].shape[0]) )], axis=0)  
        
        #Construct predictions
        predictions, predictions_prob = self.model_predict(model=lr_model.lr_model, data_columns=lr_model.data_columns, df=test)
        
        #Save predictions
        test[model_name+"predictionprob"] = [i[0] for i in predictions_prob]
        test[model_name+"prediction"] = predictions
        self.test = pd.merge(self.test, test[[model_name+"prediction", model_name+"predictionprob", 'Participant ID']], on='Participant ID', how="left")
        
        #Write metrics to file
        with open(self.evaluation_folder+"model_performance.txt", "a") as column_file:
            column_file.write("Model: "+model_name +"\n")
            column_file.write("Test size: "+str(test.shape[0]) +"\n")
            vc = self.zeror(label_column=lr_model.label_column, df=test)
            column_file.write("ZeroR: "+str(self.zeror_score(vc=vc)) +"\n")
            column_file.write("ACCURACY: "+str(self.accuracy_score(y_test=test[lr_model.label_column],
                                                               predictions=predictions)) +"\n")
            column_file.write("AUC: "+str(self.auc(y_test=test[lr_model.label_column],
                                               predictions=predictions)) +"\n")
            column_file.write("F1: "+str(self.f1_score(y_test=test[lr_model.label_column],
                                                   prediction=predictions)) +"\n\n")
        
    def lr_evaluation_copd(self, model_name, lr_model,stratisfy=True):
        '''
        This function evaluates COPD LR models, results are written to evaluations file

        Parameters
        -------
        model_name : String
            Name of model
        lr_model : LR wrapper object
        Stratisfy : Binary
            Indicates if data should be class balanced (default is True)
        '''
        #Obtain testing data
        test = self.test[lr_model.data_columns+["Participant ID", lr_model.label_column]].dropna()

        #Scaling
        if lr_model.scaler != None:
            test_id = test["Participant ID"].tolist()
            lc = test[lr_model.label_column].tolist()

            data_transformed = lr_model.scaler.transform(test[lr_model.data_columns])
            test= pd.DataFrame(data_transformed, columns=lr_model.data_columns)

            test['Participant ID'] = test_id
            test[lr_model.label_column] = lc

        #Stratification
        if stratisfy:
            test = pd.concat([test[test[lr_model.label_column]==1], test[test[lr_model.label_column]==0].sample(
                n=round(test[test[lr_model.label_column]==1].shape[0]) )], axis=0)  
        
        #Construct predictions
        predictions, predictions_prob = self.model_predict(model=lr_model.lr_model, data_columns=lr_model.data_columns, df=test)
        
        #Save predictions
        test[model_name+"predictionprob"] = [i[0] for i in predictions_prob]
        test[model_name+"prediction"] = predictions
        self.test = pd.merge(self.test, test[[model_name+"prediction", model_name+"predictionprob", 'Participant ID']], on='Participant ID', how="left")

        #Write metrics to file
        with open(self.evaluation_folder+"model_performance.txt", "a") as column_file:
            column_file.write("Model: "+model_name +"\n")
            column_file.write("Test size: "+str(test.shape[0]) +"\n")
            vc = self.zeror(label_column=lr_model.label_column, df=test)
            column_file.write("ZeroR: "+str(self.zeror_score(vc=vc)) +"\n")
            column_file.write("ACCURACY: "+str(self.accuracy_score(y_test=test[lr_model.label_column],
                                                               predictions=predictions)) +"\n")
            column_file.write("AUC: "+str(self.auc(y_test=test[lr_model.label_column],
                                               predictions=predictions)) +"\n")
            column_file.write("F1: "+str(self.f1_score(y_test=test[lr_model.label_column],
                                                   prediction=predictions)) +"\n\n")
        
    def model_predict(self,df ,model, data_columns):
        '''
        This function takes a sklearn model object and makes predictions based on supplied data

        Parameters
        -------
        df : pandas dataframe object
            Datasource
        model : sklearn model object
        data_columns : list
            List containing strings indicating which column to use from datasource

        Returns
        -------
        predictions : list
            list with binary class predictions
        predictions_prob : list
            list with probability predictions
        '''
        predictions = model.predict(df[data_columns])
        predictions_prob = model.predict_proba(df[data_columns])
        return predictions, predictions_prob
    
    def rf_evaluation_diabetes(self, rf_model, data_columns, label_column, first_occurence ,attendance_date ,model_name, cluster_model, current_cluster,stratisfy=True):#TODO: Divide into functions
        '''
        This function evaluates diabetes RF models, results are written to evaluations file

        Parameters
        -------
        model_name : String
            Name of model
        rf_model : LR wrapper object
        data_columns : list
            List containing strings indicating which column to use from datasource
        label_column : String
            String with name of the column containing labels
        first_occurence : String
            String with name of column indicating first occurence
        attendance_date : string
             String with name of column indicating attendance date
        cluster_model : cluster wrapper object
        current cluster : int
            Integer indicating the current cluster number
        Stratisfy : Binary
            Indicates if data should be class balanced (default is True)
        '''
        #Obtain test data
        test = self.test[data_columns+[label_column, attendance_date, first_occurence]]
        test_pos = test[test[label_column]==1]
        test_neg = test[(test[label_column]==0) & (test['Glycated haemoglobin (HbA1c) | Instance 0'] < 48)]
        
        test_pos = test_pos[(test_pos['Date E11 first reported (non-insulin-dependent diabetes mellitus)'] != "Code has event date matching participant's date of birth")]

        test_pos['Date of attending assessment centre | Instance 0'] = pd.to_datetime(test_pos['Date of attending assessment centre | Instance 0'])
        test_pos['Date E11 first reported (non-insulin-dependent diabetes mellitus)'] = pd.to_datetime(test_pos['Date E11 first reported (non-insulin-dependent diabetes mellitus)'])
        test_pos['diff_days'] = (test_pos[first_occurence] - test_pos[attendance_date]) / np.timedelta64(1, 'D')

        test_pos = test_pos[(test_pos['diff_days'] < rf_model.max_days) & (test_pos['diff_days'] > rf_model.min_days)]
        test_pos["LADA"] = test_pos[['Age at recruitment','Body mass index (BMI) | Instance 0', 'Glycated haemoglobin (HbA1c) | Instance 0']].apply(self.lada_alg,axis=1)
        test_pos = test_pos[test_pos["LADA"]==0]

        #Split data on sex
        if self.sex == "men":
            test_pos = test_pos[test_pos["Sex"]==1]
            test_neg = test_neg[test_neg["Sex"]==1]
        if self.sex == "women":
            test_pos = test_pos[test_pos["Sex"]==0]
            test_neg = test_neg[test_neg["Sex"]==0]

        #Remove missing data
        test_pos = test_pos[data_columns+[label_column, "Participant ID"]].dropna()
        test_neg = test_neg[data_columns+[label_column, "Participant ID"]].dropna()
        
        #Scaling
        if rf_model.scaler != None:
            lc = test_pos[label_column].tolist()
            data_transformed = rf_model.scaler.transform(test_pos[data_columns])
            test_pos = pd.DataFrame(data_transformed, columns=data_columns)
            test_pos[label_column] = lc

            lc = test_neg[label_column].tolist()
            data_transformed = rf_model.scaler.transform(test_neg[rf_model.scaler.feature_names_in_])
            test_neg = pd.DataFrame(data_transformed, columns=data_columns)
            test_neg[label_column] = lc

        #Cluster prediction
        tdc = data_columns.copy() 
        tdc.remove('Glycated haemoglobin (HbA1c) | Instance 0')

        test_pos["cluster"+model_name] = self.cluster_predict(model=cluster_model.cluster_model, data_columns=tdc, df=test_pos)
        test_pos = test_pos[test_pos["cluster"+model_name]==current_cluster]
        
        test_neg["cluster"+model_name] = self.cluster_predict(model=cluster_model.cluster_model, data_columns=tdc, df=test_neg)
        test_neg = test_neg[test_neg["cluster"+model_name]==current_cluster]

        #Save cluster prediction data
        tdf = pd.concat([test_neg[["cluster"+model_name, 'Participant ID']], test_pos[["cluster"+model_name, 'Participant ID']]]).dropna()
        self.test = self.test.merge(tdf, on='Participant ID', how="outer")
        
        #Stratification
        if stratisfy:
            test = pd.concat([test_pos, test_neg.sample(
                n=round(test_pos.shape[0]))], axis=0)
            
        else:
            test = pd.concat([test_pos, test_neg], axis=0)

        #Construct and save RF predictions
        predictions, predictions_prob = self.model_predict(model=rf_model.rf_model, data_columns=data_columns, df=test)
        test[model_name+"predictionprob"] = [i[0] for i in predictions_prob]
        test[model_name+"prediction"] = predictions
        self.test = pd.merge(self.test, test[[ model_name+"prediction", model_name+"predictionprob", 'Participant ID']], on='Participant ID', how="left")

        #Write metrics to file
        with open(self.evaluation_folder+"model_performance.txt", "a") as column_file:
            column_file.write("Model: "+model_name +"\n")
            column_file.write("Test size: "+str(test.shape[0]) +"\n")
            vc = self.zeror(label_column=label_column, df=test)
            column_file.write("ZeroR: "+str(self.zeror_score(vc=vc)) +"\n")
            column_file.write("ACCURACY: "+str(self.accuracy_score(y_test=test[label_column],
                                                               predictions=predictions)) +"\n")
            column_file.write("AUC: "+str(self.auc(y_test=test[label_column],
                                               predictions=predictions)) +"\n")
            column_file.write("F1: "+str(self.f1_score(y_test=test[label_column],
                                                   prediction=predictions)) +"\n\n")
        
    def rf_evaluation_copd(self, rf_model, data_columns, label_column ,model_name, first_occurence ,attendance_date, cluster_model, cluster_columns, current_cluster,stratisfy=True):
        '''
        This function evaluates COPD RF models, results are written to evaluations file

        Parameters
        -------
        model_name : String
            Name of model
        rf_model : LR wrapper object
        data_columns : list
            List containing strings indicating which column to use from datasource
        label_column : String
            String with name of the column containing labels
        first_occurence : String
            String with name of column indicating first occurence
        attendance_date : string
             String with name of column indicating attendance date
        cluster_model : cluster wrapper object
        current cluster : int
            Integer indicating the current cluster number
        Stratisfy : Binary
            Indicates if data should be class balanced (default is True)
        '''
        #Obtain testing data
        test = self.test[data_columns+[label_column, first_occurence ,attendance_date, 'Participant ID']]

        test_pos = test[test[label_column]==1]
        test_neg = test[(test[label_column]==0)]

        test_pos = test_pos[test_pos[first_occurence] > test_pos[attendance_date]]
        
        #Drop missing values
        test_pos = test_pos[data_columns+[label_column,'Participant ID']].dropna()
        test_neg = test_neg[data_columns+[label_column,'Participant ID']].dropna()

        #Scaling cluster
        if cluster_model.scaler != None:
            transformed = cluster_model.scaler.transform(test_pos[cluster_columns])
            temp_test_pos = pd.DataFrame(transformed, columns=cluster_columns)
            test_pos[model_name+"_cluster"] = self.cluster_predict(model=cluster_model.cluster_model, data_columns=cluster_columns, 
                                                                                    df=temp_test_pos)

            transformed = cluster_model.scaler.transform(test_neg[cluster_columns])
            temp_test_neg = pd.DataFrame(transformed, columns=cluster_columns)
            test_neg[model_name+"_cluster"] = self.cluster_predict(model=cluster_model.cluster_model, data_columns=cluster_columns,
                                                                                      df=temp_test_neg)
        
        #Add value to self.test
        tdf = pd.concat([test_neg[[model_name+"_cluster", 'Participant ID']], test_pos[[model_name+"_cluster", 'Participant ID']]])
        self.test = pd.merge(self.test, tdf, on='Participant ID',  how="outer")

        #Model scaling
        if rf_model.scaler != None:
            test_pos_id = test_pos['Participant ID'].tolist()
            lc = test_pos[label_column].tolist()
            data_transformed = rf_model.scaler.transform(test_pos[data_columns])
            test_pos = pd.DataFrame(data_transformed, columns=data_columns)
            test_pos['Participant ID'] = test_pos_id
            test_pos[label_column] = lc
            
            test_neg_id = test_neg['Participant ID'].tolist()
            lc = test_neg[label_column].tolist()
            data_transformed = rf_model.scaler.transform(test_neg[data_columns])
            test_neg = pd.DataFrame(data_transformed, columns=data_columns)
            test_neg['Participant ID'] = test_neg_id
            test_neg[label_column] = lc

        #Stratification
        if stratisfy:
            test = pd.concat([test_pos, test_neg.sample(
                n=round(test_pos.shape[0]))], axis=0)  
        else:
            test = pd.concat([test_pos, test_neg], axis=0)

        #Write metrics to file
        if test.shape[0] == 0:
            print("COPD NO MODEL FOR CLUSTER", current_cluster)
        else:
            print("COPD FOR CLUSTER: ", current_cluster)
            predictions, predictions_prob = self.model_predict(model=rf_model.rf_model, data_columns=data_columns, df=test)
            test[model_name+"_prediction"] = predictions
            test[model_name+"_predictionprob"] = [i[0] for i in predictions_prob]
            self.test = pd.merge(self.test, test[[model_name+"_prediction", model_name+"_predictionprob", 'Participant ID']], on='Participant ID', how="left")

            print("self.test_size : ", self.test.shape)

            with open(self.evaluation_folder+"model_performance.txt", "a") as column_file:
                column_file.write("Model: "+model_name +"\n")
                column_file.write("Test size: "+str(test.shape[0]) +"\n")
                vc = self.zeror(label_column=label_column, df=test)
                column_file.write("ZeroR: "+str(self.zeror_score(vc=vc)) +"\n")
                column_file.write("ACCURACY: "+str(self.accuracy_score(y_test=test[label_column],
                                                                predictions=predictions)) +"\n")
                column_file.write("AUC: "+str(self.auc(y_test=test[label_column],
                                                predictions=predictions)) +"\n")
                column_file.write("F1: "+str(self.f1_score(y_test=test[label_column],
                                                    prediction=predictions)) +"\n\n")

    def zeror(self, df,label_column):
        vc = df[label_column].value_counts()
        return vc
    
    def zeror_score(self, vc):
        return max(vc) / sum(vc)
        
    def auc(self, y_test, predictions):
        return metrics.roc_auc_score(y_test, predictions)
    
    def f1_score(self, y_test, prediction):
        return metrics.f1_score(y_test, prediction)
    
    def accuracy_score(self, y_test, predictions):
        return metrics.accuracy_score(y_test, predictions)

    def cluster_predict(self,df ,model, data_columns ):
        return model.predict(df[data_columns])

    def binary_columns_tostr(self, x):
        return "".join([str(i) for i in x])

    def linear_regression(self, x, y):
        slope, intercept, r, p, std_err = sps.linregress(x, y)
        return slope, intercept, r, p, std_err 

    def linear_model(self, x, slope, intercept):
        return slope * x + intercept
        
    def osteoporosis_modeling(self):
        '''
        This function handles the construction of the osteoporosis models
        '''

        #Column definitions
        if self.sex == 'men':
            s=1
            self.binary_columns_osteo = ['hayfever/allergic rhinitis', 
                                         'eczema/dermatitis', 
                                         'pneumonia', 
                                         'Hayfever_allergic_rhinitis_eczema_doctor', 
                                         'Brisk pace', 'Slow pace', 
                                         'Steady average pace', 
                                         'Wheeze or whistling in the chest in last year | Instance 0_No', 
                                          'hypertension', 'heart attack/myocardial infarction', 
                                          'diabetes', 'high cholesterol', 'angina', 'asthma', 
                                          'osteoarthritis', 'enlarged prostate', 'hiatus hernia', 
                                          #'unclassifiable', 
                                          'depression', 'ulcerative colitis', 
                                          'emphysema/chronic bronchitis', 'stroke', 
                                          'cataract', 'back problem', 'rheumatoid arthritis', 'epilepsy']
        else:
            s=0
            self.binary_columns_osteo = [
                'hayfever/allergic rhinitis', 'chronic sinusitis', 
                'pneumonia', 'Hayfever_allergic_rhinitis_eczema_doctor', 
                'Brisk pace', 'Slow pace', 'Steady average pace', 
                'Wheeze or whistling in the chest in last year | Instance 0_No',  
                'hypertension', 'hypothyroidism/myxoedema', 'peritonitis', 'duodenal ulcer', 
                'heart attack/myocardial infarction', 'diabetes', 'high cholesterol', 
                'fracture lower leg / ankle', 'angina', 'anxiety/panic attacks',
                'asthma', 'osteoarthritis', 'kidney stone/ureter stone/bladder stone', 
                'cholelithiasis/gall stones', 'chronic fatigue syndrome', 'psoriasis', 
                'hiatus hernia', 'heart valve problem/heart murmur', 'multiple sclerosis', 
                #'unclassifiable', 
                'allergy or anaphylactic reaction to drug', 'urinary frequency / incontinence', 
                'spine arthritis/spondylitis', 'depression', 'glaucoma', 'other renal/kidney problem', 
                'ulcerative colitis', 'ear/vestibular disorder', 'irritable bowel syndrome', 
                'colitis/not crohns or ulcerative colitis', 'emphysema/chronic bronchitis', 
                'diverticular disease/diverticulitis', 'hyperthyroidism/thyrotoxicosis', 
                'malabsorption/coeliac disease', 'stroke', 'cervical spondylosis', 
                'cataract',  'prolapsed disc/slipped disc', 
                'oesophagitis/barretts oesophagus', 'pleurisy', 'urinary tract infection/kidney infection', 
                'vaginal prolapse/uterine prolapse', 'back problem', 'essential hypertension', 
                'muscle/soft tissue problem', 'crohns disease', 'anaemia', 'ovarian cyst or cysts',
                'chronic obstructive airways disease/copd', 'heart arrhythmia', 'rheumatoid arthritis', 
                'epilepsy', 'meningitis', 'other neurological problem', 'hepatitis', 'bone disorder', 
                'gestational hypertension/pre-eclampsia', "meniere's disease", 'appendicitis', 
                'benign breast lump', 'dry eyes', 'atrial fibrillation', 'polymyalgia rheumatica', 
                'gastric/stomach ulcers', 'osteopenia', 'rectal or colon adenoma/polyps', 
                'helicobacter pylori', 'eye/eyelid problem', 'parkinsons disease', 'joint disorder', 
                'varicose veins', 'fracture wrist / colles fracture', 'rheumatic fever', 
                'systemic lupus erythematosis/sle', 'pernicious anaemia', "sjogren's syndrome/sicca syndrome"
            ]
        
        #Linear regression dataprep
        x = [i for i in range(140,190)]

        y = []
        for i in x:
            t = self.train[(self.train['Standing height | Instance 0']==i) & (self.train['Date M81 first reported (osteoporosis without pathological fracture)_binary']==0) & (self.train["Sex"] ==s)]
            to = self.train[(self.train['Standing height | Instance 0']==i) & (self.train['Date M81 first reported (osteoporosis without pathological fracture)_binary']==1)& (self.train["Sex"] ==s)]
            if to.shape[0] == 0:
                y.append(0)
            else:
                y.append(to.shape[0]/(t.shape[0]+to.shape[0])*100)

        #Model construction
        self.slope_osteo, self.intercept_osteo, r, p, std_err = self.linear_regression(x=x, y=y)
        self.nb_osteo.nb_model = self.naive_bayes(df=self.train, data_columns=self.binary_columns_osteo, label_column='Date M81 first reported (osteoporosis without pathological fracture)_binary')

    def evaluation_osteoporosis(self, label_column='Date M81 first reported (osteoporosis without pathological fracture)_binary'):
        '''
        This function handles the evaluation of the osteoporosis models

        Parameters
        -------
        label_column : String
            String indicating the name of the label column
            (default = Date M81 first reported (osteoporosis without pathological fracture)_binary)
        '''
        
        #Construct and save predicions
        predictions, predictions_prob = self.model_predict(model=self.nb_osteo.nb_model, data_columns=self.binary_columns_osteo, df=self.test)
        predictions_prob = [l[0] for l in predictions_prob]
        self.test['perc_nb_osteo'] = predictions_prob
        self.test['perc_linreg_osteo'] = self.test['Standing height | Instance 0'].apply(self.linear_model,slope=self.slope_osteo, intercept=self.intercept_osteo )

        #Write metrics to file
        with open(self.evaluation_folder+"model_performance.txt", "a") as column_file:
            column_file.write("OSTEOPOROSIS"+"\n")

            column_file.write("Model: NAIVE BAYES"+"\n")
            column_file.write("Test size: "+str(self.test.shape[0]) +"\n")
            column_file.write("Mean: "+str(self.test['perc_nb_osteo'].dropna().mean())+"\n")
            column_file.write("Median: "+str(self.test['perc_nb_osteo'].dropna().median())+"\n")
            column_file.write("Max: "+str(self.test['perc_nb_osteo'].dropna().max())+"\n")
            column_file.write("Min: "+str(self.test['perc_nb_osteo'].dropna().min())+"\n")
            vc = self.zeror(label_column=label_column, df=self.test)
            column_file.write("ZeroR: "+str(self.zeror_score(vc=vc)) +"\n")
            column_file.write("ACCURACY: "+str(self.accuracy_score(y_test=self.test[label_column],
                                                               predictions=predictions)) +"\n")
            column_file.write("AUC: "+str(self.auc(y_test=self.test[label_column],
                                               predictions=predictions)) +"\n")
            column_file.write("F1: "+str(self.f1_score(y_test=self.test[label_column],
                                                   prediction=predictions)) +"\n")

            column_file.write("MODEL: REGRESSION"+"\n")
            column_file.write("Mean: "+str(self.test['perc_linreg_osteo'].mean())+"\n")
            column_file.write("Median: "+str(self.test['perc_linreg_osteo'].median())+"\n")
            column_file.write("Max: "+str(self.test['perc_linreg_osteo'].max())+"\n")
            column_file.write("Min: "+str(self.test['perc_linreg_osteo'].min())+"\n\n")



    def asthma_modeling(self):
        '''
        This function handles the construction of the asthma models
        ''' 
        #Column definition
        if self.sex == 'men':
            s=1
            self.binary_columns_asthma = [
                'hayfever/allergic rhinitis',
                'pneumonia',
                'Hayfever_allergic_rhinitis_eczema_doctor',
                'Brisk pace',
                'Slow pace',
                'Wheeze or whistling in the chest in last year | Instance 0_No',
                'hypertension',
                'hypothyroidism/myxoedema',
                'heart attack/myocardial infarction',
                'diabetes',
                'high cholesterol',
                'angina',
                'osteoarthritis',
                'gout',
                'enlarged prostate',
                'hiatus hernia',
                'depression',
                'irritable bowel syndrome',
                'emphysema/chronic bronchitis',
                'stroke',
                'cataract',
                'prolapsed disc/slipped disc',
                'back problem',
                'eczema/dermatitis'
            ]
        else:
            s=0
            self.binary_columns_asthma = [
                'hayfever/allergic rhinitis',
                'pneumonia',
                'Hayfever_allergic_rhinitis_eczema_doctor',
                'Brisk pace',
                'Slow pace',
                'Steady average pace',
                'Wheeze or whistling in the chest in last year | Instance 0_No',
                'hypertension',
                'hypothyroidism/myxoedema',
                'heart attack/myocardial infarction',
                'diabetes',
                'high cholesterol',
                'angina',
                'anxiety/panic attacks',
                'osteoarthritis',
                'cholelithiasis/gall stones',
                'psoriasis',
                'hiatus hernia',
                'migraine',
                'allergy or anaphylactic reaction to drug',
                'spine arthritis/spondylitis',
                'depression',
                'glaucoma',
                'sciatica',
                'allergy/hypersensitivity/anaphylaxis',
                'irritable bowel syndrome',
                'emphysema/chronic bronchitis',
                'diverticular disease/diverticulitis',
                'hyperthyroidism/thyrotoxicosis',
                'stroke',
                'cervical spondylosis',
                'cataract',
                'osteoporosis',
                'prolapsed disc/slipped disc',
                'endometriosis',
                'vaginal prolapse/uterine prolapse',
                'back problem',
                'muscle/soft tissue problem',
                'anaemia',
                'eczema/dermatitis',
                'rheumatoid arthritis',
                'joint disorder'
            ]

        #Linear regression dataprep
        x = [i for i in range(140,190)]

        y = []
        for i in x:
            t = self.train[(self.train['Pack years of smoking']==i) & (self.train['all_asthma_binary']==0) & (self.train["Sex"] ==s)]
            to = self.train[(self.train['Pack years of smoking']==i) & (self.train['all_asthma_binary']==1)& (self.train["Sex"] ==s)]
            if to.shape[0] == 0:
                y.append(0)
            else:
                y.append(to.shape[0]/(t.shape[0]+to.shape[0])*100)

        #Construct models
        self.slope_asthma, self.intercept_asthma, r, p, std_err = self.linear_regression(x=x, y=y)
        self.nb_atshma.nb_model = self.naive_bayes(df=self.train, data_columns=self.binary_columns_asthma, label_column='all_asthma_binary')
        
    def evaluation_asthma(self,label_column = 'all_asthma_binary'):
        '''
        This function handles the evaluation of the asthma models

        Parameters
        -------
        label_column : String
            String indicating the name of the label column
            (default = all_asthma_binary)
        '''
        #Constructing and saving predictions
        predictions, predictions_prob = self.model_predict(model=self.nb_atshma.nb_model, data_columns=self.binary_columns_asthma, df=self.test)
        predictions_prob = [l[0] for l in predictions_prob]
        self.test['perc_nb_asthma'] = predictions_prob
        self.test['perc_linreg_asthma'] = self.test['Standing height | Instance 0'].apply(self.linear_model,slope=self.slope_asthma, intercept=self.intercept_asthma)

        #Write metrics to file
        with open(self.evaluation_folder+"model_performance.txt", "a") as column_file:
            column_file.write("ASTHMA"+"\n")
            column_file.write("Model: NAIVE BAYES"+"\n")
            column_file.write("Test size: "+str(self.test.shape[0]) +"\n")
            column_file.write("Mean: "+str(self.test['perc_nb_asthma'].mean())+"\n")
            column_file.write("Median: "+str(self.test['perc_nb_asthma'].median())+"\n")
            column_file.write("Max: "+str(self.test['perc_nb_asthma'].dropna().max())+"\n")
            column_file.write("Min: "+str(self.test['perc_nb_asthma'].dropna().min())+"\n")
            vc = self.zeror(label_column=label_column, df=self.test)
            column_file.write("ZeroR: "+str(self.zeror_score(vc=vc)) +"\n")
            column_file.write("ACCURACY: "+str(self.accuracy_score(y_test=self.test[label_column],
                                                               predictions=predictions)) +"\n")
            column_file.write("AUC: "+str(self.auc(y_test=self.test[label_column],
                                               predictions=predictions)) +"\n")
            column_file.write("F1: "+str(self.f1_score(y_test=self.test[label_column],
                                                   prediction=predictions)) +"\n")

            column_file.write("Regression percentage:"+"\n")
            column_file.write("Mean: "+str(self.test['perc_linreg_asthma'].mean())+"\n")
            column_file.write("Median: "+str(self.test['perc_linreg_asthma'].median())+"\n")
            column_file.write("Max: "+str(self.test['perc_linreg_asthma'].max())+"\n")
            column_file.write("Min: "+str(self.test['perc_linreg_asthma'].min())+"\n\n")

    def naive_bayes(self, df, data_columns, label_column):
        model = GaussianNB()
        model.fit(df[data_columns], df[label_column])
        return model
    

class controller():
    def __init__(self, wd, evaluation_folder,file=None, path=None, suppress_warnings=False, stratisfy=True) :
        #DATA PREPROCESSING
        self.file = file
        self.evaluation_folder = evaluation_folder
        self.path = path
        self.wd = wd
        self.stratisfy = stratisfy

        self.dp = dataPreprocessing(wd=self.wd)

        if suppress_warnings:
            warnings.filterwarnings('ignore')

        if self.file == None:
            print("Constructing dataframe")
            self.data_preprocessing()
            self.df = self.dp.df
            
            if path != None:
                print("Saving dataframe to csv")
                self.df.to_csv(self.path)
        else:
            print("Loading from file")
            self.df = self.load_file()

        print("Save columns")
        self.columns_to_file()

        print("Split in test/train")
        self.split_test_train()
        print("Test size: ", self.test.shape)
        #MODELING

        #Both
        print("Model construction: combined")
        self.mc = modelConstruction(df=self.df, train=self.train, 
                                    test=self.test, 
                                    evaluation_folder=self.evaluation_folder,
                                    dp=self.dp)
        self.copd_model(self.mc)
        self.copd_model_list = [self.mc.emphysema_model_current,
                                self.mc.other_copd_model_current,
                                self.mc.chronic_bronchitis_model_current,
                                self.mc.emphysema_model_past,
                                self.mc.other_copd_model_past
                                ]
        self.lr_evaluation_copd_con(construction_obj=self.mc, model_list=self.copd_model_list)

        self.cvd_model(self.mc)
        self.cvd_model_list = [self.mc.cvd_hf_model,
                               self.mc.cvd_isch_model]
        self.lr_evaluation_cvd_con(construction_obj=self.mc, model_list=self.cvd_model_list)

        #Men only
        print("Model construction: men only")
        self.mc_men = modelConstruction(df=self.df[self.df["Sex"]==1], 
                                        test=self.test[self.test["Sex"]==1],
                                        train=self.train[self.train["Sex"]==1],
                                        evaluation_folder=self.evaluation_folder,
                                        dp=self.dp,
                                        sex="men")
        self.diabetes_model(self.mc_men)
        
        self.diabetes_men_model_list = [self.mc_men.niddm_na_one_lada_rfmodel_one,
                                    self.mc_men.niddm_na_one_lada_rfmodel_two ,
                                    self.mc_men.niddm_na_one_lada_rfmodel_three ,
                                    self.mc_men.niddm_na_one_lada_rfmodel_four ,
                                    self.mc_men.niddm_na_five_lada_rfmodel_one ,
                                    self.mc_men.niddm_na_five_lada_rfmodel_two ,
                                    self.mc_men.niddm_na_five_lada_rfmodel_three ,
                                    self.mc_men.niddm_na_five_lada_rfmodel_four ,
                                    self.mc_men.niddm_na_fiveten_lada_rfmodel_one ,
                                    self.mc_men.niddm_na_fiveten_lada_rfmodel_two ,
                                    self.mc_men.niddm_na_fiveten_lada_rfmodel_three ,
                                    self.mc_men.niddm_na_fiveten_lada_rfmodel_four ,
                                    self.mc_men.niddm_na_ten_lada_rfrmodel_one ,
                                    self.mc_men.niddm_na_ten_lada_rfrmodel_two ,
                                    self.mc_men.niddm_na_ten_lada_rfrmodel_three ,
                                    self.mc_men.niddm_na_ten_lada_rfrmodel_four ]
        self.diabetes_men_cluster_list = [self.mc_men.niddm_na_one_lada_clustermodel,
                                            self.mc_men.niddm_na_five_lada_clustermodel,
                                            self.mc_men.niddm_na_fiveten_lada_clustermodel,
                                            self.mc_men.niddm_na_ten_lada_clustermodel]
        self.rf_evaluation_diabetes_con(construction_obj=self.mc_men, model_list=self.diabetes_men_model_list, cluster_list=self.diabetes_men_cluster_list)
        
        self.osteoporosis_model(construction_obj=self.mc_men)
        self.mc_men.evaluation_osteoporosis()

        self.asthma_model(construction_obj=self.mc_men)
        self.mc_men.evaluation_asthma()

        #Women only
        print("Model construction: women only")
        self.mc_women = modelConstruction(df=self.df[self.df["Sex"]==0], 
                                        test=self.test[self.test["Sex"]==0],
                                        train=self.train[self.train["Sex"]==0],
                                        evaluation_folder=self.evaluation_folder,
                                        dp=self.dp,
                                        sex="women")
        self.diabetes_model(self.mc_women)

        self.diabetes_women_model_list = [self.mc_women.niddm_na_one_lada_rfmodel_one,
                                    self.mc_women.niddm_na_one_lada_rfmodel_two ,
                                    self.mc_women.niddm_na_one_lada_rfmodel_three ,
                                    self.mc_women.niddm_na_one_lada_rfmodel_four ,
                                    self.mc_women.niddm_na_five_lada_rfmodel_one ,
                                    self.mc_women.niddm_na_five_lada_rfmodel_two ,
                                    self.mc_women.niddm_na_five_lada_rfmodel_three ,
                                    self.mc_women.niddm_na_five_lada_rfmodel_four ,
                                    self.mc_women.niddm_na_fiveten_lada_rfmodel_one ,
                                    self.mc_women.niddm_na_fiveten_lada_rfmodel_two ,
                                    self.mc_women.niddm_na_fiveten_lada_rfmodel_three ,
                                    self.mc_women.niddm_na_fiveten_lada_rfmodel_four ,
                                    self.mc_women.niddm_na_ten_lada_rfrmodel_one ,
                                    self.mc_women.niddm_na_ten_lada_rfrmodel_two ,
                                    self.mc_women.niddm_na_ten_lada_rfrmodel_three ,
                                    self.mc_women.niddm_na_ten_lada_rfrmodel_four ]
        self.diabetes_women_cluster_list = [self.mc_women.niddm_na_one_lada_clustermodel,
                                            self.mc_women.niddm_na_five_lada_clustermodel,
                                            self.mc_women.niddm_na_fiveten_lada_clustermodel,
                                            self.mc_women.niddm_na_ten_lada_clustermodel]
        self.rf_evaluation_diabetes_con(construction_obj=self.mc_women, model_list=self.diabetes_women_model_list, cluster_list=self.diabetes_women_cluster_list)
        
        self.osteoporosis_model(construction_obj=self.mc_women)
        self.mc_women.evaluation_osteoporosis()

        self.asthma_model(construction_obj=self.mc_women)
        self.mc_women.evaluation_asthma()

        #Save results to csv
        print("Saving results")
        results = pd.concat([self.mc_men.test,self.mc_women.test])
        results = results.merge(self.mc.test[["Participant ID"]+self.mc.test.columns.tolist()[self.df.shape[1]:]])
        print('Results shape: ', results.shape)
        results.to_csv("results")

    def data_preprocessing(self):
        self.dp.factory()

    def split_test_train(self):
        self.train = self.df.sample(frac = 0.80)
        self.test = self.df.drop(self.train.index)

    def load_file(self):
        return pd.read_csv(self.file, low_memory=False)
    
    def diabetes_model(self, construction_obj):
        print("DIABETES MODELING")
        construction_obj.diabetes_modeling()
    
    def copd_model(self, construction_obj):
        print("COPD MODELING")
        construction_obj.copd_modeling()
    
    def osteoporosis_model(self, construction_obj):
        print('OSTEOPOROSIS MODELING')
        construction_obj.osteoporosis_modeling()

    def cvd_model(self, construction_obj):
        print('CVD MODELING')
        construction_obj.cvd_modeling()

    def asthma_model(self, construction_obj):
        print('ASTHMA MODELING')
        construction_obj.asthma_modeling()
    
    def columns_to_file(self): 
        with open(self.evaluation_folder+"columns.txt", "w") as column_file:
            for i in self.df.columns.to_list():
                column_file.write(i+"\n")

    def rf_evaluation_diabetes_con(self, construction_obj, model_list, cluster_list):
        x = 1
        y =0
        cc = 0
        for model in model_list:
            construction_obj.rf_evaluation_diabetes_rebuilt(rf_model=model, 
                                            model_name=str(x)+"_Diabetes_"+construction_obj.sex+"_"+str(model.min_days)+"_"+str(model.max_days)+"_"+str(cc),
                                           data_columns=construction_obj.diabetes_data_columns,
                                           label_column='Date E11 first reported (non-insulin-dependent diabetes mellitus)_onehot',
                                           first_occurence='Date E11 first reported (non-insulin-dependent diabetes mellitus)',
                                           attendance_date='Date of attending assessment centre | Instance 0',
                                           cluster_model=cluster_list[y],
                                           current_cluster=cc,
                                           stratisfy=self.stratisfy)
            
            if x % 4==0 and x !=0:
                y+=1
            if cc == 3:
                cc-=4
            cc+=1
            x+=1
            
    def rf_evaluation_copd_con_deprecated(self, construction_oj, model_list):
        cc=0
        for model in model_list:
            construction_oj.rf_evaluation_copd(rf_model=model, model_name="COPD_"+str(cc), 
                                               data_columns=construction_oj.copd_data_columns, 
                                               label_column='Date J43 first reported (emphysema)_binary' , 
                                               first_occurence = "Date J44 first reported (other chronic obstructive pulmonary disease)",
                                               attendance_date = 'Date of attending assessment centre | Instance 0', 
                                               cluster_model = construction_oj.copd_clustermodel, 
                                               cluster_columns = construction_oj.copd_cluster_columns, 
                                               current_cluster = cc)
            cc+=1

    def lr_evaluation_copd_con(self, construction_obj, model_list):
        x = 0
        for model in model_list:
            if x <= 2:
                construction_obj.lr_evaluation_copd( model_name=model.label_column+"_current", 
                                               lr_model=model,stratisfy=self.stratisfy )
            else:
                construction_obj.lr_evaluation_copd( model_name=model.label_column+"_past", 
                                               lr_model=model, stratisfy=self.stratisfy)
            x+=1

    def lr_evaluation_cvd_con(self, construction_obj, model_list):
        x = 0
        modelnames = ['Date I50 first reported (heart failure)'
            ,'Date I25 first reported (chronic ischaemic heart disease)']
        for model in model_list:
            construction_obj.lr_evaluation_copd( model_name=modelnames[x], 
                                               lr_model=model,stratisfy=self.stratisfy )
            x+=1
        

#Driver code
if __name__ == "__main__":
    con = controller("C:/Users/keimp/", file="C:/Users/keimp/NHS/dataframe.csv",
                     evaluation_folder="C:/Users/keimp/NHS/Code/experimental_modeling/Meta_learner/evaluations/",
                        suppress_warnings=True,
                        stratisfy=True)

    
    
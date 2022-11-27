from typing import List
from fastapi import FastAPI
from pydantic import BaseModel

import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from numpy import dot
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity
import uvicorn

app = FastAPI()

# 데이터 전처리

class DataPreprocessing(object):
   
    def ListToArray(self, row_data): 
        self.arr = np.array(row_data)
                                    
    def MakeColumns(self):
        self.user_id = []
        for i in range(self.arr.shape[0]):
            self.user_id.append(self.arr[i][0])
            
        self.channels = []
        for i in range(self.arr.shape[0]):
            self.channels.append(self.arr[i][1])
            
        self.cate = []
        for i in range(self.arr.shape[0]):
            self.cate.append(self.arr[i][2:])
    
    def ArrayToDataframe(self):
        self.row_df = pd.DataFrame({'Channels': self.channels, 'Category': self.cate}, index=self.user_id)
    
    def Extraction(self):
        self.category = []
        
        for i in range(self.row_df['Category'].shape[0]):
            np.array(pd.DataFrame(self.row_df['Category'][i]).value_counts()).reshape(1,-1)
            
            self.category_series = pd.DataFrame(self.row_df['Category'][i]).value_counts()
            
            self.df_category = pd.DataFrame(np.array(self.category_series).reshape(1, -1), 
                               columns = [j for j in self.category_series.index])

            self.category.append(self.df_category.sort_values(by=0, axis=1).iloc[0, -2:].index.tolist())
            
        self.first_category = pd.DataFrame(self.category)[0].str[0]
        self.second_category = pd.DataFrame(self.category)[1].str[0]
    
    def ToDf(self):
        df = pd.DataFrame({'channels': self.channels, 'first_category': self.first_category.values, 
                           'second_category': self.second_category.values}, index=self.user_id)
        
        return df


# sklearn의 cosine_similarity 사용
# 코사인 유사도 계산 시 시간복잡도 문제 해결

class Classification(object):
    
    def __init__(self, df_data):
        self.df = df_data
        self.n_samples = self.df.shape[0]
    
    def Ohe(self, df): # 각 샘플에 대해 추출한 가장 선호하는 2개의 특성에 대해 원핫인코딩
        ohe = OneHotEncoder(sparse=False)
        self.train_feature = ohe.fit_transform(np.array(df[['first_category', 'second_category']]).reshape(-1,1))
    
    def makeData(self, channels):
        channels = np.array(channels).reshape(self.n_samples, -1)
        self.X_train = np.hstack((channels, np.array(self.train_feature).reshape(self.n_samples, -1)))
        
    def Kmeans(self): # k_means 클러스터링
        model = KMeans(n_clusters=2)
        model.fit(self.X_train)
        return np.array(model.predict(self.X_train))
    
    def ReturnResult_cosine_similarity(self, k=1):
        
        k_users = np.empty((0,k))
        for i in range(0, self.X_train.shape[0]):
            distance = [cosine_similarity(self.X_train[i].reshape(1,-1), self.X_train)]
            
            # i 번째 데이터 포인터와 다른 데이터들 사이의 거리를 작은순으로 정렬한 후 해당 인덱스를 이용해서 가장 가까운 k명의 이용자 선별
            distance_idx = np.array(distance).ravel().argsort()
            k_users = np.append(k_users, [distance_idx[:k]], axis=0)
                    
        return k_users
        

class reqBody(BaseModel):
    youtubeSubscriptionData : List[List[str]]

# class channelIds(BaseModel):


# class Dataset(BaseModel):
#     kind: str
#     etag: str
#     id: str
#     topicDetails: channelIds




@app.get("/")
async def root():
    return {"message": "Hello World"}

# @app.get("/youtube/get-subscription")
# async def get_data(item: Dataset):
#     topicId = item.topicDetails
#     return topicId



@app.post("/ml/match")
async def match_making(data: reqBody):

    row_data = data.youtubeSubscriptionData
    dt = DataPreprocessing()
    dt.ListToArray(row_data)
    dt.MakeColumns()
    dt.ArrayToDataframe()
    dt.Extraction()
    df = dt.ToDf()
    cl = Classification(df)
    cl.Ohe(df)
    cl.makeData(dt.channels)
    k_users = cl.ReturnResult_cosine_similarity()
    
    return {"message": "Hello World"}
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
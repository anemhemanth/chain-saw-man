import seaborn as sb
val=sb.load_dataset("titanic")
val.head()
survived	pclass	sex	age	sibsp	parch	fare	embarked	class	who	adult_male	deck	embark_town	alive	alone
0	0	3	male	22.0	1	0	7.2500	S	Third	man	True	NaN	Southampton	no	False
1	1	1	female	38.0	1	0	71.2833	C	First	woman	False	C	Cherbourg	yes	False
2	1	3	female	26.0	0	0	7.9250	S	Third	woman	False	NaN	Southampton	yes	True
3	1	1	female	35.0	1	0	53.1000	S	First	woman	False	C	Southampton	yes	False
4	0	3	male	35.0	0	0	8.0500	S	Third	man	True	NaN	Southampton	no	True
val.drop("deck",axis=1,inplace=True)
val['age'].fillna(24,inplace=True)
val.dropna(inplace=True)
val.info()
<class 'pandas.core.frame.DataFrame'>
Index: 889 entries, 0 to 890
Data columns (total 14 columns):
 #   Column       Non-Null Count  Dtype   
---  ------       --------------  -----   
 0   survived     889 non-null    int64   
 1   pclass       889 non-null    int64   
 2   sex          889 non-null    object  
 3   age          889 non-null    float64 
 4   sibsp        889 non-null    int64   
 5   parch        889 non-null    int64   
 6   fare         889 non-null    float64 
 7   embarked     889 non-null    object  
 8   class        889 non-null    category
 9   who          889 non-null    object  
 10  adult_male   889 non-null    bool    
 11  embark_town  889 non-null    object  
 12  alive        889 non-null    object  
 13  alone        889 non-null    bool    
dtypes: bool(2), category(1), float64(2), int64(4), object(5)
memory usage: 86.1+ KB
val.drop("who",axis=1,inplace=True)
 
from sklearn.preprocessing import LabelEncoder
model=LabelEncoder()
val["class"].unique()
['Third', 'First', 'Second']
Categories (3, object): ['First', 'Second', 'Third']
model.fit(['First','Second','Third'])

LabelEncoder
LabelEncoder()
val["class"]=model.transform(val["class"])#'3'(2)-'1'(0)-'2'(1)
val.embarked=model.fit_transform(val.embarked)#'S'('2')-"C"-0
val.adult_male=model.fit_transform(val.adult_male)
val.head()
survived	pclass	sex	age	sibsp	parch	fare	embarked	class	adult_male	embark_town	alive	alone
0	0	3	male	22.0	1	0	7.2500	2	2	1	Southampton	no	False
1	1	1	female	38.0	1	0	71.2833	0	0	0	Cherbourg	yes	False
2	1	3	female	26.0	0	0	7.9250	2	2	0	Southampton	yes	True
3	1	1	female	35.0	1	0	53.1000	2	0	0	Southampton	yes	False
4	0	3	male	35.0	0	0	8.0500	2	2	1	Southampton	no	True
val.alone=model.fit_transform(val["alone"])
def func(val):
  if val=="no":
    return 0
  else:
    return 1
val.alive=val.alive.apply(func)
def func1(val):
  if val=="female":
    return 0
  else:
    return 1
val.sex=val.sex.apply(func)
val.embark_town=model.fit_transform(val.embark_town)
val.head()
survived	pclass	sex	age	sibsp	parch	fare	embarked	class	adult_male	embark_town	alive	alone
0	0	3	1	22.0	1	0	7.2500	2	2	1	2	0	0
1	1	1	1	38.0	1	0	71.2833	0	0	0	0	1	0
2	1	3	1	26.0	0	0	7.9250	2	2	0	2	1	1
3	1	1	1	35.0	1	0	53.1000	2	0	0	2	1	0
4	0	3	1	35.0	0	0	8.0500	2	2	1	2	0	1
import matplotlib.pyplot as plt
fig=plt.figure(figsize=(15,10))
ax=fig.add_axes([1,1,1,1])
sb.heatmap(data=val.corr(),annot=True,cmap="rainbow",ax=ax)
<Axes: >

 
fig

y=val.alive
val.drop(["alive","survived"],axis=1,inplace=True)
from sklearn.model_selection import train_test_split
xtra,xtest,ytra,ytest=train_test_split(val,y,test_size=0.2)
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(xtra,ytra)

RandomForestClassifier
RandomForestClassifier()
model.score(xtest,ytest)
0.8033707865168539
 
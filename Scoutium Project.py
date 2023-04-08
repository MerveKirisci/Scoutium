# Scoutium Project

# İş Problemi
# Scout’lar tarafından izlenen futbolcuların özelliklerine verilen puanlara göre, oyuncuların hangi sınıf (average, highlighted) oyuncu olduğunu tahminleme.

# Veri Seti Hikayesi
# Veri seti Scoutium’dan maçlarda gözlemlenen futbolcuların özelliklerine göre scoutların değerlendirdikleri futbolcuların, maç içerisinde puanlanan özellikleri ve puanlarını içeren bilgilerden oluşmaktadır.



import numpy as np
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, cross_val_score, validation_curve
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Görünüm ayarları
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# Görevler
#Adım 1: scoutium_attributes.csv ve scoutium_potential_labels.csv dosyalarını okutunuz.

scoutium_attributes = pd.read_csv("datasets/scoutium_attributes.csv", sep=';')
scoutium_potential_labels = pd.read_csv("datasets/scoutium_potential_labels.csv", sep=';')

# Adım 2: Okutmuş olduğumuz csv dosyalarını merge fonksiyonunu kullanarak birleştiriniz.
# ("task_response_id", 'match_id', 'evaluator_id' "player_id" 4 adet değişken üzerinden birleştirme işlemini gerçekleştiriniz.
new_scout = pd.merge(scoutium_attributes, scoutium_potential_labels, how="right", on=["task_response_id", "match_id", "evaluator_id", "player_id"])
new_scout.head()

df = new_scout.copy()
df.head()

# Adım 3: position_id içerisindeki Kaleci (1) sınıfını veri setinden kaldırınız.
df.drop(df[df['position_id'] == 1].index, inplace = True)
df[["position_id"]].value_counts()

# position_id
# 2              1972
# 6              1428
# 10             1088
# 8              1020
# 3               986
# 7               986
# 4               884
# 9               850
# 5               816

# Adım 4: potential_label içerisindeki below_average sınıfını veri setinden kaldırınız.( below_average sınıfı tüm verisetinin %1'ini oluşturur)
df.drop(df[df['potential_label'] == 'below_average'].index, inplace=True)
df[["potential_label"]].value_counts()

# potential_label
# average            7922
# highlighted        1972


# Adım 5: Oluşturduğunuz veri setinden “pivot_table” fonksiyonunu kullanarak bir tablo oluşturunuz. Bu pivot table'da her satırda bir oyuncu olacak şekilde manipülasyon yapınız.

# 1: İndekste “player_id”,“position_id” ve “potential_label”, sütunlarda “attribute_id” ve değerlerde scout’ların oyunculara verdiği puan “attribute_value” olacak şekilde pivot table’ı oluşturunuz.

sc_pivot = pd.pivot_table(data=df, index=['player_id', 'position_id', 'potential_label'], columns=['attribute_id'], values='attribute_value')
sc_pivot

# 2: “reset_index” fonksiyonunu kullanarak indeksleri değişken olarak atayınız ve “attribute_id” sütunlarının isimlerini stringe çeviriniz.

sc_pivot.reset_index(inplace=True)
sc_pivot = sc_pivot.astype(str)
sc_pivot


# Adım 6: Label Encoder fonksiyonunu kullanarak “potential_label” kategorilerini (average, highlighted) sayısal olarak ifade ediniz.

def label_encoder(df, column):
    labelencoder = LabelEncoder()
    df[column] = labelencoder.fit_transform(df[column])
    return df

sc_pivot = label_encoder(sc_pivot, 'potential_label')
sc_pivot.head()

sc_pivot.columns.astype(str)
sc_pivot.columns

# Adım 7: Sayısal değişken kolonlarını “num_cols” adıyla bir listeye atayınız.
def grab_col_names(dataframe, cat_th=10, car_th=20):

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = sc_pivot.columns[3:]
num_cols
# num_cols olarak sayısal değişkenleri listeye atarken 3. indexten sonrakileri almamızın nedeni
# sc_pivot columnlarının ilk 3 değerinin sayısal olmaması, devamının sayısal olmasıdır.

# Adım 8: Kaydettiğiniz bütün “num_cols” değişkenlerindeki veriyi ölçeklendirmek için StandardScaler uygulayınız.

sscaler = StandardScaler()
sc_pivot[num_cols] = sscaler.fit_transform(sc_pivot[num_cols])
sc_pivot.head()


# Adım 9: Elimizdeki veri seti üzerinden minimum hata ile futbolcuların potansiyel etiketlerini tahmin eden bir makine öğrenmesi modeli geliştiriniz.
# (Roc_auc, f1, precision, recall, accuracy metriklerini yazdırınız.)

# For modelling our data set, first of all we'll apply cross validation
y = sc_pivot["potential_label"]
X = sc_pivot.drop(["potential_label", "player_id"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=45)


#BASE MODEL

models = [('LR', LinearRegression()),
          ("Ridge", Ridge()),
          ("Lasso", Lasso()),
          ("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor())]

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

# RMSE: 0.375 (LR)
# RMSE: 0.3716 (Ridge)
# RMSE: 0.4048 (Lasso)
# RMSE: 0.4048 (ElasticNet)
# RMSE: 0.3831 (KNN)
# RMSE: 0.4352 (CART)
# RMSE: 0.316 (RF)
# RMSE: 0.3215 (GBM)
# RMSE: nan (XGBoost)
# RMSE: nan (LightGBM)

rf_model = RandomForestClassifier(random_state=17)

rf_model.get_params()


cv_results = cross_validate(rf_model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

# 0.87
# 0.56
# 0.89

rf_params = {"max_depth": [5, 8, None],
             "max_features": [3, 5, 7, "auto"],
             "min_samples_split": [2, 5, 8, 15, 20],
             "n_estimators": [100, 200, 500]}

rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

rf_best_grid.best_params_

rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(rf_final, X, y, cv=10
                            , scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

# 0.88
# 0.59
# 0.90

#########################################
# Analyzing Model Complexity with Learning Curves (BONUS)
#########################################

def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    train_score, test_score = validation_curve(
        model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score,
             label="Training Score", color='b')

    plt.plot(param_range, mean_test_score,
             label="Validation Score", color='g')

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show(block=True)


rf_val_params = [["max_depth", [5, 8, 15, 20, 30, None]],
                 ["max_features", [3, 5, 7, "auto"]],
                 ["min_samples_split", [2, 5, 8, 15, 20]],
                 ["n_estimators", [10, 50, 100, 200, 500]]]


rf_model = RandomForestClassifier(random_state=17)

for i in range(len(rf_val_params)):
    val_curve_params(rf_model, X, y, rf_val_params[i][0], rf_val_params[i][1])

rf_val_params[0][1]

# Adım 10: Değişkenlerin önem düzeyini belirten feature_importance fonksiyonunu kullanarak özelliklerin sıralamasını çizdiriniz.
def plot_importance(model, features, num=len(X), save=True):
   feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
   plt.figure(figsize=(10, 10))
   sns.set(font_scale=1)
   sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                    ascending=False)[0:num])
   plt.title('Features')
   plt.tight_layout()
   plt.show()
   if save:
       plt.savefig('importances.png')


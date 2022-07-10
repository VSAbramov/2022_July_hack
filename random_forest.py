from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder

from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.over_sampling import SMOTENC
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
import catboost

from dateutil import parser

def str_to_time(df):
    for col in time_cols:
        obs_num = df[col].shape[0]
        for i in range(obs_num):
            idx = df.index[i]
            df.loc[idx, col] = parser.parse(df.loc[idx, col])
    return df

def time_to_secs(df):
    zero_time = parser.parse('00:00:00')
    obs_num = df.shape[0]

    for col in time_cols:
        for i in range(obs_num):
            idx = df.index[i]
            df.loc[idx, col] = (df.loc[idx, col] - zero_time).seconds
    
    return df   

def prepare_data(df):
    # заменить na в числовых переменных (мешает переводу в категории)
    cols_to_replace_na = cat_ord_cols + num_cols + time_cols
    for col in cols_to_replace_na:
        df[col].fillna(value=0, inplace = True)

    # Национальность: вопрос больше похожи на русских или наоборот
    # ВИЧ/СПИД: мало данных
    # Статус Курения: дублирует "Возраст курения"
    # 'Пассивное курение': дублирует 'Частота пасс кур'
    # 'Время засыпания'   :   надо будет разобраться как её пользоваться
    # 'Время пробуждения' : надо будет разобраться как её пользоваться
    columns_to_drop = ['Национальность', 
                       'ВИЧ/СПИД', 
                       'Статус Курения', 
                       'ID_y',
                       'Пассивное курение',
                       #'Религия'
                       ]
    #columns_to_drop += time_cols

    used_columns = []
    for col in df.columns:
        if not col in columns_to_drop:
            used_columns.append(col)
    df = df.loc[:,used_columns]

    # перевести время в число
    df.loc[:, time_cols] = str_to_time(df.loc[:, time_cols])
    df.loc[:, time_cols] = time_to_secs(df.loc[:, time_cols])
    # global time_cals
    # for col in time_cals:
    #     df[col] = df.loc[:, col].astype('str')
    #     df[col] = time.fromisoformat(df[col])
    
    #df.drop(columns_to_drop, axis=1, inplace = True)

    # под вопросом, нужно ли это делать (возможно стоит)
    # df['Семья'].replace({'раздельное проживание (официально не разведены)':'в разводе'}, inplace=True)
    # df['Образование'].replace({'2 - начальная школа':'3 - средняя школа / закон.среднее / выше среднего'}, inplace=True)
    # df['Профессия'].replace({'вооруженные силы':'служащие'}, inplace=True)

    df['Возраст курения'].fillna(value = 0, inplace = True)
    df['Сигарет в день'].fillna(value = 0, inplace = True)
    df['Частота пасс кур'].fillna(value = 0, inplace = True)
    df['Возраст алког'].fillna(value = 0, inplace = True)

    df.dropna(axis=0, inplace = True)
    #df.fillna(0, inplace=True)
    #df.fillna(value=0, inplace=True)
    df.set_index('ID', inplace=True)

    #global cat_ord_cols
    cat_ord_attribs = intersection(cat_ord_cols, df.columns)
    df = encode_cols(df, cat_ord_attribs)
    return df

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def chose_model(X, y, iter_num, hyper_params, rand_state):
    pipe = make_pipeline(RandomForestClassifier(n_jobs=-1, 
                                                random_state = rand_state))
    model = RandomizedSearchCV(pipe, 
                               hyper_params, 
                               n_iter=iter_num, 
                               cv=5, 
                               scoring = 'f1',
                               random_state = rand_state)
    model.fit(X, y)
    return model
    

binar_cols = ['Пол', 
        'Вы работаете?', 
        'Выход на пенсию',
        'Прекращение работы по болезни',
        'Сахарный диабет',
        'Гепатит',
        'Онкология',
        'Хроническое заболевание легких',
        'Бронжиальная астма',
        'Туберкулез легких',
        'ВИЧ/СПИД',
        'Регулярный прим лекарственных средств',
        'Травмы за год',
        'Переломы',
        'Пассивное курение',
        'Сон после обеда',
        'Спорт, клубы',
        'Религия, клубы',
        'Артериальная гипертензия',
        'ОНМК',
        'Стенокардия, ИБС, инфаркт миокарда',
        'Сердечная недостаточность',
        'Прочие заболевания сердца'
        ]

cat_unord_cols = ['Семья', 
        'Этнос',
        'Национальность',
        'Религия',
        'Профессия',
        ]

cat_ord_cols = ['Образование',
        'Статус Курения',
        'Частота пасс кур',
        'Алкоголь']

num_cols = ['Возраст курения',
        'Сигарет в день',
        'Возраст алког',
        ]

time_cols = ['Время засыпания', 
        'Время пробуждения']

def encode_cols(df, cols):
    change_dict = {}
    for col in cols:
        if col == 'Образование':
            change_dict = {'2 - начальная школа': 0,
                        '3 - средняя школа / закон.среднее / выше среднего': 1,
                        '4 - профессиональное училище': 2,
                        '5 - ВУЗ': 3}
        elif col == 'Статус Курения':
            change_dict = {'Никогда не курил(а)': 0,
                           'Бросил(а)': 1,
                           'Курит': 2}
        elif col == 'Частота пасс кур':
            change_dict = {'1-2 раза в неделю': 1,
                           '3-6 раз в неделю': 2,
                           'не менее 1 раза в день': 3,
                           '2-3 раза в день': 4,
                           '4 и более раз в день': 5}
        elif col == 'Алкоголь':
            change_dict = {'никогда не употреблял': 0, 
                           'ранее употреблял': 1,
                           'употребляю в настоящее время': 2}
        df[col].replace(change_dict, inplace=True)
    return df


def resample_score(X, y, hyper_params, rand_state, cat_cols_bool):
    # print(hyper_params)
    # hyper_params = dict()
    # for key, value in hyper_params_raw.items():
    #     hyper_params[key] = value[0]

    # cat_cols_bool = []
    # for i in range(X.shape[1]):
    #     cat_cols_bool.append(i<X.shape[1]-3)

    splitter = StratifiedShuffleSplit(n_splits = 5, 
                                      test_size = 0.2, 
                                      random_state = rand_state)

    score = []

    for train_idx, test_idx in splitter.split(X, y):
        X_train, y_train = X[train_idx], y.iloc[train_idx]
        X_test, y_test = X[test_idx], y.iloc[test_idx]

        smt = SMOTENC(categorical_features = cat_cols_bool, 
                      random_state = rand_state)
        X_resample, y_resample = smt.fit_resample(X_train, y_train)

        model = RandomForestClassifier(n_jobs=-1, 
                                       random_state = rand_state, 
                                       **hyper_params)
        model.fit(X_resample, y_resample)

        y_hat = model.predict(X_test)

        score.append(f1_score(y_test, y_hat))

    score = np.array(score)
    return score.mean()

    
def resample_score_boost(X, y, hyper_params, rand_state, cat_cols_bool):
    splitter = StratifiedShuffleSplit(n_splits = 5, 
                                      test_size = 0.2, 
                                      random_state = rand_state)

    score = []

    for train_idx, test_idx in splitter.split(X, y):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        smt = SMOTENC(categorical_features = cat_cols_bool, 
                      random_state = rand_state)
        X_resample, y_resample = smt.fit_resample(X_train, y_train)

        DF_rsm = pd.DataFrame(X_resample)
        DF_rsm.iloc[:,0:54] = (DF_rsm.iloc[:,0:54]).astype(int).astype(str)
        DF_rsm.iloc[:,-1] = (DF_rsm.iloc[:,-1]).astype(int).astype(str)
        
        model = catboost.CatBoostClassifier(verbose=False, 
                                            random_state = rand_state,
                                            cat_features=list(range(54)).append(DF_rsm.shape[1]-1),
                                            **hyper_params)
        model.fit(X_resample, y_resample)

        y_hat = model.predict(X_test)


        TP = np.logical_and(y_hat==1,y_test==1).sum()
        FN = np.logical_and(y_hat==0,y_test==1).sum()
        FP = np.logical_and(y_hat==1,y_test==0).sum()

        # print(f'TP: {round(TP, 2)}')
        # print(f'FP: {round(FP, 2)}')
        # print(f'FN: {round(FN, 2)}')

        if TP + FP == 0:
            pre = 0
        else:
            pre = TP/(TP + FP)
        if TP + FN == 0:
            rec = 0
        else:
            rec = TP/(TP + FN)

        if pre + rec == 0:
            f1 = 0
        else:
            f1 = 2* (pre * rec) / (pre + rec)

        score.append(f1)

    score = np.array(score)
    return score.mean()
import pandas
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from textblob import TextBlob, Word

from Helpers.data_prep import *
from nltk.sentiment import SentimentIntensityAnalyzer
import seaborn as sns

import warnings

from pandas.core.common import SettingWithCopyWarning
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

dataset = pd.read_csv('DS/v2_cleaned_chatlogs.csv')
dataset = pd.read_csv('DS/chatlogs.csv')
dataset.head(10)

dataset = delete_columns(dataset,
                         ['association_to_offender', 'case_total_reports', 'allied_report_count', 'enemy_report_count',
                          'most_common_report_reason'])
lower_letters(dataset, 'message')
lower_letters(dataset, 'champion_name')
stop_words_eng = stop_words_creator('english')
remove_stop_words(dataset, 'message', stop_words_eng)
remove_punctuations(dataset, 'message')
remove_numbers(dataset, 'message')
remove_rare_words(dataset, 'message', 4)
remove_champ_names(dataset, 'message')
remove_gaming_acronyms(dataset, 'message')
remove_na_msgs(dataset, 'message')
dataset.to_csv('v2_cleaned_chatlogs.csv')
dataset.drop(dataset[dataset['message'].map(len) < 5].index,
             inplace=True)  # mesaj karakter sayısı x olanlardan küçük olanları temizler

dataset['message'] = dataset['message'].astype(str)
dataset['message'].len().mean()
dataset.reset_index()
sns.boxplot(x=dataset['message'].str.len())
plt.show()

#########################################################################################
# Count Vector (Frekans temsilleri)

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

y = dataset['toxic_language_label']
X = dataset['message']

vectorizer = CountVectorizer()
X_count = vectorizer.fit_transform(X)

X_c = vectorizer.fit_transform(dataset['message'])
vectorizer.get_feature_names()
X_c.toarray()

#########################################################################################
# TF-IDF

tf_idf_word_vectorizer = TfidfVectorizer()
X_tf_idf_word = tf_idf_word_vectorizer.fit_transform(X)

#########################################################################################
# Modelleme

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import classification_report

# WORD
X_train, X_test, y_train, y_test = train_test_split(X_count,
                                                    y,
                                                    test_size=0.20, random_state=17)

# TF-IDF
X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(X_tf_idf_word,
                                                                            y,
                                                                            test_size=0.20, random_state=17)

# Logistic Regression -----------------------------------------------------------------------
# Logistic Regression Count Vector
log_model = LogisticRegression(solver='liblinear',class_weight='balanced').fit(X_train, y_train)
cv_result_word = cross_validate(log_model,
                                X_train,
                                y_train,
                                scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
                                cv=10)

# {'fit_time': array([0.81169915, 0.77016234, 0.77917051, 0.7646575 , 0.75615072,
#         0.79818702, 0.85273314, 0.82020617, 0.84923053, 0.88976574]),
#  'score_time': array([0.03252697, 0.02301979, 0.02101755, 0.02101851, 0.02051711,
#         0.02302003, 0.02352023, 0.02151847, 0.02352047, 0.02351999]),
#  'test_accuracy': array([0.95900687, 0.96021682, 0.96007163, 0.96162037, 0.96210435,
#         0.95963605, 0.9607008 , 0.9583777 , 0.95968444, 0.96152171]),
#  'test_precision': array([0.87398447, 0.87959927, 0.87911888, 0.88175368, 0.88212859,
#         0.87742873, 0.88039323, 0.87317073, 0.87841281, 0.88254605]),
#  'test_recall': array([0.97014028, 0.96773547, 0.96773547, 0.97134269, 0.97315167,
#         0.96814266, 0.9689441 , 0.96834302, 0.96694049, 0.96973948]),
#  'test_f1': array([0.91955551, 0.92156489, 0.92130115, 0.92438257, 0.92540726,
#         0.9205563 , 0.92254865, 0.91829755, 0.92055317, 0.92409052]),
#  'test_roc_auc': array([0.98315397, 0.98393885, 0.98204642, 0.98413135, 0.98501019,
#         0.98368532, 0.98409792, 0.98321352, 0.98370538, 0.98539734])}

lrcv_params_def = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [1000, 2000],
                   "colsample_bytree": [0.5, 0.7, 1]}

lrcv_best_grid = GridSearchCV(log_model,
                              lrcv_params_def,
                              cv=5,
                              n_jobs=-1,
                              verbose=True).fit(X_train, X_test)

cv_result_word['test_accuracy'].mean()  # 0.9602940737668547
cv_result_word["test_precision"].mean()  # 0.8788536446467482
cv_result_word["test_recall"].mean()  # 0.969221532626463
cv_result_word['test_f1'].mean()  # 0.9218257561793803
cv_result_word['test_roc_auc'].mean()  # 0.9838380265959517

#                  precision  recall   f1-score  support
#            0       0.99      0.96      0.97     39282
#            1       0.88      0.97      0.92     12373
#     accuracy                           0.96     51655
#    macro avg       0.93      0.96      0.95     51655
# weighted avg       0.96      0.96      0.96     51655

y_pred_word = log_model.predict(X_test)
print(classification_report(y_test, y_pred_word))

acc = round(accuracy_score(y_test, y_pred_word), 2)
cm = confusion_matrix(y_test, y_pred_word)
sns.heatmap(cm, annot=True, fmt=".0f")
plt.xlabel('y_pred_word')
plt.ylabel('y_test')
plt.title('Karmaşıklık Matrisi'.format(acc), size=10)
plt.show()

dataset.groupby("champion_name").agg({"toxic_language_label": "mean", }).sort_values(by="toxic_language_label",
                                                                                     ascending=False).to_csv('toxic_champs.csv')

# ----------------------------------------------------------------------------------------
# Logistic Regression TF-IDF
log_model_tf_idf = LogisticRegression(solver='liblinear', class_weight='balanced').fit(X_train_tfidf, y_train_tfidf)
cv_result_tf_idf = cross_validate(log_model_tf_idf,
                                  X_train_tfidf,
                                  y_train_tfidf,
                                  scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
                                  cv=10)

# {'fit_time': array([0.54446816, 0.48691893, 0.52044749, 0.49842763, 0.49242306,
#         0.47390747, 0.5704906 , 0.47390747, 0.54346752, 0.48341537]),
#  'score_time': array([0.0280242 , 0.02351975, 0.02452111, 0.02301955, 0.0205183 ,
#         0.02201891, 0.01901627, 0.02402115, 0.02151895, 0.02702308]),
#  'test_accuracy': array([0.9584261 , 0.95915207, 0.95881328, 0.96041042, 0.96118478,
#         0.95920046, 0.95973284, 0.95668377, 0.95881328, 0.9610861 ]),
#  'test_precision': array([0.87878232, 0.88374676, 0.88161534, 0.88458702, 0.88622534,
#         0.88181149, 0.88473636, 0.87633223, 0.88404453, 0.88816766]),
#  'test_recall': array([0.96032064, 0.95671343, 0.95811623, 0.96152305, 0.96293328,
#         0.95972751, 0.95812462, 0.95551994, 0.95471849, 0.95971944]),
#  'test_f1': array([0.91774394, 0.91878368, 0.91827523, 0.92145189, 0.92298829,
#         0.91912117, 0.91996922, 0.91421451, 0.91802331, 0.92255827]),
#  'test_roc_auc': array([0.98469269, 0.98462674, 0.98342854, 0.98555611, 0.98592056,
#         0.98466605, 0.98552568, 0.98464384, 0.98474703, 0.9865449 ])}

cv_result_tf_idf['test_accuracy'].mean()  # 0.9593503101640899
cv_result_tf_idf["test_precision"].mean()  # 0.8830049037550799
cv_result_tf_idf["test_recall"].mean()  # 0.9587416628488393
cv_result_tf_idf['test_f1'].mean()  # 0.9193129523289608
cv_result_tf_idf['test_roc_auc'].mean()  # 0.9850352140726881

y_pred_tf_idf = log_model.predict(X_train_tfidf)
print(classification_report(y_train_tfidf, y_pred_tf_idf))

#                  precision  recall   f1-score  support
#            0       0.93      0.98      0.96    156714
#            1       0.94      0.76      0.84     49905
#     accuracy                           0.93    206619
#    macro avg       0.93      0.87      0.90    206619
# weighted avg       0.93      0.93      0.93    206619

#########################################################################################

### Random Forest / Count Vectors -------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier

rf_model_word = RandomForestClassifier(class_weight='balanced').fit(X_train, y_train)

rf_cv_result_word = cross_validate(rf_model_word,
                                   X_train,
                                   y_train,
                                   scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
                                   cv=5,
                                   n_jobs=-1)

# {'fit_time': array([587.24825621, 584.01197195, 587.37936974, 587.28278613,
#         581.35768843]),
#  'score_time': array([10.85133648, 10.9259007 , 10.87135386, 10.85684109, 10.98044682]),
#  'test_accuracy': array([0.97638176, 0.97679315, 0.97817249, 0.97524441, 0.9759698 ]),
#  'test_precision': array([0.94060084, 0.94044132, 0.94370052, 0.93868756, 0.94110718]),
#  'test_recall': array([0.96302976, 0.96503356, 0.96733794, 0.96022443, 0.96062519]),
#  'test_f1': array([0.95168317, 0.95257875, 0.95537305, 0.94933386, 0.95076603]),
#  'test_roc_auc': array([0.99434563, 0.9949509 , 0.99588272, 0.99515993, 0.99433659])}

rf_cv_result_word['test_accuracy'].mean()  # 0.9765123219910894
rf_cv_result_word["test_precision"].mean()  # 0.9409074851591329
rf_cv_result_word["test_recall"].mean()  # 0.963250175333133
rf_cv_result_word['test_f1'].mean()  # 0.963250175333133
rf_cv_result_word['test_roc_auc'].mean()  # 0.9949351535991011

rf_y_pred_word = rf_model_word.predict(X_test)
print(classification_report(y_test, rf_y_pred_word))

#                  precision  recall  f1-score  support
#            0       0.99      0.98      0.99     39282
#            1       0.94      0.97      0.95     12373
#     accuracy                           0.98     51655
#    macro avg       0.96      0.97      0.97     51655
# weighted avg       0.98      0.98      0.98     51655

#########################################################################################

### Random Forest / TF_IDF Word-Level ---------------------------------------------------

rf_model_tf_idf = RandomForestClassifier().fit(X_train_tfidf, y_train_tfidf)
rf_cv_result_tf_idf = cross_validate(rf_model_tf_idf,
                                     X_train_tfidf,
                                     y_train_tfidf,
                                     scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
                                     cv=5,
                                     n_jobs=-1)

# {'fit_time': array([540.69420266, 539.50668073, 538.30114365, 537.58953023,
#         532.93702769]),
#  'score_time': array([10.53206086, 10.65766907, 10.8333199 , 10.82381344, 11.01797938]),
#  'test_accuracy': array([0.97461524, 0.97582519, 0.97638176, 0.97444584, 0.97524381]),
#  'test_precision': array([0.93401361, 0.93450077, 0.93599303, 0.93103448, 0.93476995]),
#  'test_recall': array([0.96292957, 0.9677387 , 0.96844004, 0.9657349 , 0.96483318]),
#  'test_f1': array([0.9482512 , 0.95082935, 0.95194012, 0.94806728, 0.94956367]),
#  'test_roc_auc': array([0.99483048, 0.99506806, 0.99578104, 0.9953893 , 0.99488505])}

rf_cv_result_tf_idf['test_accuracy'].mean()  # 0.975302367843437
rf_cv_result_tf_idf["test_precision"].mean()  # 0.9340623675524986
rf_cv_result_tf_idf["test_recall"].mean()  # 0.96593527702635
rf_cv_result_tf_idf['test_f1'].mean()  # 0.949730324726531
rf_cv_result_tf_idf['test_roc_auc'].mean()  # 0.9951907846090651

rf_y_pred_tf_idf = rf_model_tf_idf.predict(X_train_tfidf)
print(classification_report(y_train_tfidf, rf_y_pred_tf_idf))

#                  precision  recall   f1-score  support
#            0       1.00      1.00      1.00    156714
#            1       1.00      1.00      1.00     49905
#     accuracy                           1.00    206619
#    macro avg       1.00      1.00      1.00    206619
# weighted avg       1.00      1.00      1.00    206619


#########################################################################################

# LightGBM / Count Vector----------------------------------------------------------------
# X_train, X_test, y_train, y_test
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV

# X_train, X_test, y_train, y_test

X_train_lgbm = X_train.astype('float32')
X_text_lgbm = X_test.astype('float32')
y_train_lgbm = y_train.astype('float32')
y_test_lgbm = y_test.astype('float32')

lgbm_model_count_vector = LGBMClassifier(class_weight="balanced", random_state=17).fit(X_train_lgbm, y_train_lgbm)
lgbm_model_count_vector.get_params()

lgbm_cv_results_count_vec = cross_validate(lgbm_model_count_vector,
                                           X_train_lgbm,
                                           y_train_lgbm,
                                           scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
                                           cv=5,
                                           n_jobs=-1)

# {'fit_time': array([5.24351025, 5.27453732, 5.28104305, 5.21398544, 5.1309135 ]),
#  'score_time': array([1.21154284, 1.25558138, 1.24407077, 1.19402671, 1.20553732]),
#  'test_accuracy': array([0.94823831, 0.94983545, 0.95135998, 0.95073081, 0.94973743]),
#  'test_precision': array([0.9311634 , 0.93383805, 0.93278315, 0.93146519, 0.93247975]),
#  'test_recall': array([0.84841198, 0.85272017, 0.86063521, 0.85923254, 0.85372207]),
#  'test_f1': array([0.8878637 , 0.89143755, 0.89525795, 0.89389202, 0.89136461]),
#  'test_roc_auc': array([0.97265664, 0.97281278, 0.97320382, 0.97256766, 0.97161665])}

lgbm_params_def = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [1000, 2000],
                   "colsample_bytree": [0.5, 0.7, 1]}

lgbm_best_grid = GridSearchCV(lgbm_model_count_vector,
                              lgbm_params_def,
                              cv=5,
                              n_jobs=-1,
                              verbose=True).fit(X_train_lgbm, y_train_lgbm)

lgbm_cvec_final = lgbm_model_count_vector.set_params(**lgbm_best_grid.best_params_,
                                                     random_state=17).fit(X_train_lgbm, y_train_lgbm)

final_lgbm_cv_results_count_vec = cross_validate(lgbm_cvec_final,
                                                 X_train_lgbm,
                                                 y_train_lgbm,
                                                 scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
                                                 cv=5,
                                                 n_jobs=-1)

final_lgbm_cv_results_count_vec['test_accuracy'].mean()  # 0.9749635752242428
final_lgbm_cv_results_count_vec["test_precision"].mean()  # 0.9468057205892579
final_lgbm_cv_results_count_vec["test_recall"].mean()  # 0.9497044384330227
final_lgbm_cv_results_count_vec['test_f1'].mean()  # 0.9482505927808779
final_lgbm_cv_results_count_vec['test_roc_auc'].mean()  # 0.9836344044392027

lgm_y_pred_countvec = lgbm_model_count_vector.predict(X_train_lgbm)
print(classification_report(y_train_lgbm, lgm_y_pred_countvec))

#                  precision  recall   f1-score  support
#          0.0       0.99      0.99      0.99    156714
#          1.0       0.97      0.97      0.97     49905
#     accuracy                           0.98    206619
#    macro avg       0.98      0.98      0.98    206619
# weighted avg       0.98      0.98      0.98    206619

# LightGBM / TF-IDF----------------------------------------------------------------

# X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf

X_train_lgbm, X_test_lgbm, y_train_lgbm, y_test_lgbm = train_test_split(dataset['message'],
                                                                        dataset['toxic_language_label'],
                                                                        test_size=0.20,
                                                                        random_state=17,
                                                                        shuffle=True)

tf_idf_word_vectorizer = TfidfVectorizer()
X_train_lgbm_tfidf = tf_idf_word_vectorizer.fit_transform(X_train_lgbm)
X_test_lgbm_tfidf = tf_idf_word_vectorizer.fit_transform(X_test_lgbm)

f32_X_train_lgbm_tfidf = X_train_lgbm_tfidf.astype('float32')
f32_X_test_lgbm_tfidf = X_test_lgbm_tfidf.astype('float32')
f32_y_train_lgbm_tfidf = y_train_lgbm.astype('float32')
f32_y_test_lgbm_idf = y_test_lgbm.astype('float32')

lgbm_model_tfidf = LGBMClassifier().fit(f32_X_train_lgbm_tfidf, f32_X_test_lgbm_tfidf)
lgbm_model_count_vector.get_params()

# ----------------------------------------------------------------------------------------------------------------------

# Hiperparametre Optimizasyonu
from sklearn.model_selection import GridSearchCV

rf_model = RandomForestClassifier(random_state=17)

rf_params = {"max_depth": [8, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [2, 5, 8],
             "n_estimators": [100, 200]}

rf_best_grid = GridSearchCV(rf_model,
                            rf_params,
                            cv=5,
                            n_jobs=-1,
                            verbose=True).fit(X_count, y)

rf_best_grid.best_params_  # {'max_depth': None,
#  'max_features': 'auto',
#  'min_samples_split': 5,
#  'n_estimators': 200}

rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X_count, y)  # 0.9785568289908608



from io import StringIO
import csv
import os
from collections import Counter
import pandas as pd
warnings.filterwarnings("ignore")

### file name of data in database
file_info   = "INFO.csv"
file_dass   = 'dass.csv'
file_altman = 'altman.csv'

### file name of users' information
file_user_p = "id_user.xlsx"
file_account= "accounts.csv"

### directories
dir_raw_p   = "./raw/"
dir_patient = "p/"
dir_split   = "./split_p/"
dir_info    = "info/"


####################################################################
####################################################################
####################################################################

### create info directory
d = dir_split + dir_info
try:
    if not os.path.exists(d):
        os.makedirs(d)
except OSError:
    print ('Error: Creating directory. ' +  d)

### tags of data types
# 0=文字
# 1=語音
# 2=純情緒標記
# 3=影片
# 4=每日情緒
# 5=睡眠時間_1
# 6=每周自評量表_1
# 7=每周自評量表_2
# 8=睡眠時間_2

str_type = ["文字", "語音", "純情緒標記", "影片", "每日情緒", "睡眠時間1",
     "每周自評量表1", "每周自評量表2", "睡眠時間2"]

# create directory by data type
for i in range(0, len( str_type)):
    d = dir_split + dir_info + str(i)
    try:
        if not os.path.exists(d):
            os.makedirs(d)
    except OSError:
        print ('Error: Creating directory. ' +  d)

####################################################################
####################################################################
####################################################################

###  Read info csv file
data_info = pd.read_csv( dir_raw_p + file_info,  encoding="utf8")

### Read account csv file
account = pd.read_csv( dir_raw_p + dir_patient + file_account, dtype=str)
# change type to string
account[ 'Account']  = account[ 'Account'].astype( str) 
account[ 'mobile']   = account[ 'mobile'].astype( str) 


#### Read user_id csv file
# 0 user
# 1 type
# 19 date
id_user = pd.read_excel( dir_raw_p + dir_patient + file_user_p, header = None, dtype=str)
# change type to string
id_user[ 1] = id_user[ 1].astype( str)
# remove the space
for i in range( len( id_user)):
    id_user[ 1].iloc[ i] = id_user[ 1].iloc[ i].strip()


####################################################################
####################################################################
####################################################################
### insert phone to info

# get account's phone
acc_p = {}
# get all account
acc = Counter( data_info[ 'Account'].tolist())
for n in acc.keys():
    # get index in registration
    tmp = account[ account[ 'Account'] == n].index
    if len( tmp) > 0:
        # get phone
        acc_p[ n] = account[ 'mobile'].iloc[ tmp[0]]
    else:
        # no phone number
        acc_p[ n] = 'NAN'


### 
tmp = ['NAN'] * len( data_info)
for i in range( len( data_info)):
    name    = data_info[ 'Account'].iloc[i]
    tmp[ i] = acc_p[ name]


# insert
data_info[ 'phone'] = tmp


# Count how many users
phones = data_info[ 'phone'].tolist()
c_phone = Counter( phones)


#### get info
# 0=文字
# 1=語音
# 2=純情緒標記
# 3=Video
# 4=每日情緒
# 5=起床時間
# 6=每周自評量表
# 7=每周自評量表
# 8=睡覺時間
# Count of types
t = data_info[ 'type'].tolist()
c_type = Counter(t)


####################################################################
####################################################################
####################################################################

#### Split data by info type
datas = data_info.values.tolist()
#
for types in c_type.keys():
    od = dir_split + dir_info
    # split data by type
    t = []
    for row in datas:
        if row[1] == types:
            t.append(row)
    # split data by user
    for p in c_phone.keys():
        of = od + str( types) + '/' + p + '.csv'
        with open( of, 'w', newline='', encoding="utf8") as csvfile:
            writer = csv.writer(csvfile)
            for row in t:
                if row[ 28] == p :
                    n = writer.writerow(row)



####################################################################
####################################################################
####################################################################
#### Split data by user

# create directory
d = dir_split + dir_info + 'user/'
try:
    if not os.path.exists(d):
        os.makedirs(d)
except OSError:
    print ('Error: Creating directory. ' +  d)


# Split data by user
for p in c_phone.keys():
    of = dir_split + dir_info + "user/" + p + ".csv"
    # split data by user
    with open(of, 'w', newline='', encoding="utf8") as csvfile:
        writer = csv.writer(csvfile)
        for row in datas:
            if row[ 28] == p:
                n = writer.writerow(row)








### 客觀情緒
# ##########################################################################
# ##########################################################################
# ##########################################################################
# ##########################################################################

# file_T  = "TER.xlsx"
# file_S  = "SER.xlsx"
# file_V  = "VER.xlsx"


# ### TER text
# #  Read file
# data_T = pd.read_excel( dir_raw_p + file_T, dtyp=str)
# data_T[ 'Account'] = data_T[ 'Account'].astype(str)
# #### Read account
# account = pd.read_csv( dir_raw_p + dir_patient + file_account, dtype=str)
# account[ 'Account']  = account[ 'Account'].astype( str) 
# account[ 'mobile']  = account[ 'mobile'].astype( str) 

# #### Read user_id
# id_user = pd.read_excel( dir_raw_p + dir_patient + file_user_p, header = None, dtype=str)
# id_user[ 1] = id_user[ 1].astype( str) 
# for i in range( len( id_user)):
#     id_user[ 1].iloc[ i] = id_user[ 1].iloc[ i].strip()


# ####
# data_T[ 'phone'] = 'NAN'
# for i in range( len( data_T)):
#     name    = data_T[ 'Account'].iloc[i]
#     idx     = account[ account[ 'Account'] == str( name)].index
#     if len( idx) > 0:
#         phone = account[ 'mobile'].iloc[ idx[0]]
#         data_T[ 'phone'].iloc[ i] = phone


# # Count how many users
# phones = data_T[ 'phone'].tolist()
# c_phone = Counter( phones)



# datas = data_T.values.tolist()
# # Split data by user
# for p in c_phone.keys():
#     of = dir_split + dir_info + "user/" + p + "_TER.csv"
#     # split data by user
#     with open(of, 'w', newline='', encoding="utf8") as csvfile:
#         writer = csv.writer(csvfile)
#         for row in datas:
#             if row[ 7] == p:
#                 n = writer.writerow(row)






# ### SER speech
# #  Read file
# data_S = pd.read_excel( dir_raw_p + file_S, dtyp=str)
# data_S[ 'Account'] = data_S[ 'Account'].astype(str)
# #### Read account
# account = pd.read_csv( dir_raw_p + dir_patient + file_account, dtype=str)
# account[ 'Account']  = account[ 'Account'].astype( str) 
# account[ 'mobile']  = account[ 'mobile'].astype( str) 

# #### Read user_id
# id_user = pd.read_excel( dir_raw_p + dir_patient + file_user_p, header = None, dtype=str)
# id_user[ 1] = id_user[ 1].astype( str) 
# for i in range( len( id_user)):
#     id_user[ 1].iloc[ i] = id_user[ 1].iloc[ i].strip()


# ####
# data_S[ 'phone'] = 'NAN'
# for i in range( len( data_S)):
#     name    = data_S[ 'Account'].iloc[i]
#     idx     = account[ account[ 'Account'] == str( name)].index
#     if len( idx) > 0:
#         phone = account[ 'mobile'].iloc[ idx[0]]
#         data_S[ 'phone'].iloc[ i] = phone


# # Count how many users
# phones = data_S[ 'phone'].tolist()
# c_phone = Counter( phones)



# datas = data_S.values.tolist()
# # Split data by user
# for p in c_phone.keys():
#     of = dir_split + dir_info + "user/" + p + "_SER.csv"
#     # split data by user
#     with open(of, 'w', newline='', encoding="utf8") as csvfile:
#         writer = csv.writer(csvfile)
#         for row in datas:
#             if row[ 6] == p:
#                 n = writer.writerow(row)






# ##########################################################################
# ##########################################################################
# ##########################################################################
# ##########################################################################


# # Count how many users
# phones = data_info[ 'phone'].tolist()
# c_phone = Counter( phones)



# for u in c_phone.keys():
#     user = u
#     #
#     dr =  pd.read_csv( dir_split + dir_info + "user/" + u + ".csv", header=None, index_col=False)
#     contents = dr[ 2]
#     print( user)
#     #
#     Anger   = [ 0.0] * len( dr)
#     Happy   = [ 0.0] * len( dr)
#     Neutral = [ 0.0] * len( dr)
#     Sad     = [ 0.0] * len( dr)
#     #### TER
#     idx_T =  dr.index[ dr[ 1] == 0].tolist()
#     len( idx_T)
#     if len( idx_T) > 0 :
#         try:
#             dT = pd.read_csv( dir_split + dir_info + "user/" + user  + "_TER.csv", header=None, index_col=False)
#         except:
#             pass
#         #
#         for i in idx_T :
#             text = contents[ i ]
#             idx  = dT.index[ dT[ 1] == text]
#             if len( idx) > 0 :
#                 Anger[ i]   = dT[ 3].iloc[ idx].values[0]
#                 Happy[ i]   = dT[ 4].iloc[ idx].values[0]
#                 Neutral[ i] = dT[ 5].iloc[ idx].values[0]
#                 Sad[ i]     = dT[ 6].iloc[ idx].values[0]
#             else:
#                 print( 'zero')
#     #### SER
#     idx_S =  dr.index[ dr[ 1] == 1].tolist()
#     if len( idx_S) > 0 :
#         try:
#             dS = pd.read_csv( dir_split + dir_info + "user/" + user + "_SER.csv", header=None, index_col=False)
#         except:
#             pass
#         #
#         for i in idx_S :
#             speech  = contents[ i ]
#             idx     = dS.index[ dS[ 1] == speech]
#             if len( idx) > 0 :
#                 Anger[ i]   = dS[ 3].iloc[ idx].values[0]
#                 Happy[ i]   = dS[ 4].iloc[ idx].values[0]
#                 Sad[ i]     = dS[ 5].iloc[ idx].values[0]
#             else:
#                 print( 'zero')
#     ####
#     dr[ 'Anger']    = Anger
#     dr[ 'Happy']    = Happy
#     dr[ 'Neutral']  = Neutral
#     dr[ 'Sad']      = Sad
#     #
#     of = dir_split + dir_info + "user/" + str( user) + "_mix.csv"
#     dr.to_csv( of, index=False)




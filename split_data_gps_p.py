
from io import StringIO
import csv
import os
from collections import Counter
import pandas as pd

### file name of data in database
# file_gps   = ["GPS_1.csv", "GPS_2.csv", "GPS_3.csv", "GPS_4.csv"]
file_gps   = ["GPS_1.csv", "GPS_2.csv", "GPS_3.csv","GPS_4.csv","GPS_5.csv","GPS_6.csv","GPS_7.csv"]

### file name of users' information
file_user_p = "id_user.xlsx"
file_account= "accounts.csv"

### directories
dir_raw_p   = "./raw/"
dir_patient = "p/"
dir_split   = "./split_p/"
dir_gps     = "gps/"


### create directory
d = dir_split + dir_gps
try:
    if not os.path.exists(d):
        os.makedirs(d)
except OSError:
    print ('Error: Creating directory. ' +  d)



####################################################################
####################################################################
####################################################################

### Read gps files
#
index_col   = ''
data_gps    = []
for i in file_gps:
    with open( dir_raw_p + i, newline='') as csvfile:
        tmp = list( csv.reader(csvfile))
    print( len( tmp))
    tmp = tmp[ 1:]
    index_col = tmp[0]
    data_gps = data_gps + tmp

# list to pd
data_gps = pd.core.frame.DataFrame( data_gps)



### Read account
account = pd.read_csv( dir_raw_p + dir_patient + file_account, dtype=str)
# change type to string
account[ 'Account']  = account[ 'Account'].astype( str) 
account[ 'mobile']  = account[ 'mobile'].astype( str) 


### Read user_id
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
acc = Counter( data_gps[ 0].tolist())
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
tmp = ['NAN'] * len( data_gps)
for i in range( len( data_gps)):
    name    = data_gps[ 0].iloc[i]
    tmp[ i] = acc_p[ name]



# insert phone
data_gps[ 'phone'] = tmp


####################################################################
####################################################################
####################################################################

# Count how many users & phone
phones = data_gps[ 'phone'].tolist()
c_phone = Counter( phones)


# Split data by user phone
od = dir_split + dir_gps
for p in c_phone.keys():
    of = od + p + '.csv'
    data_gps[ data_gps['phone']==p].to_csv( of, index=0)


####################################################################



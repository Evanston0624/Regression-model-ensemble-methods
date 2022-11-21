

from io import StringIO
import csv
import os
import datetime
import math
from os import listdir
from os.path import isfile, join
from haversine import haversine
from scipy.signal import lombscargle
from sklearn import preprocessing
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from math import sin, cos, sqrt, atan2, radians
from sklearn.cluster import DBSCAN
import sklearn.cluster as skc
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
count_dis = 0



head = ["Account", "startlat", "startlng", "endlat", "endlng", "starttime", 
        "endtime", "costtime",  "distance",  "formatetime",  "speed", "offline", "phone"]
#Account  startlat  startlng  endlat  endlng  starttime  endtime  
# costtime  distance  formatetime  speed

##
dir_split   = "./split_p/"
dir_raw     = "./raw/"
dir_gps     = "gps/"
dir_gps_week= "gps_week/"
dir_train   = "train_p/"


################################################################################
def get_gps_by_userdate( name, date):
    dr =  pd.read_csv(dir_split + dir_gps + name + ".csv", index_col=False)
    dateFormatter = "%Y-%m-%d %H:%M:%S"
    dr['formatetime'] = pd.to_datetime(dr['formatetime']).dt.strftime( dateFormatter)
    dr['formatetime'] = dr['formatetime'].astype('datetime64[ns]')
    # get the first Monday
    gap     = ( 7 - dr['formatetime'][0].weekday()) % 7
    start   = dr['formatetime'][0].replace(hour=0, minute=0, second=0) + datetime.timedelta( days = gap)
    # get the weekdate we want
    target_date  = datetime.datetime.strptime( date, dateFormatter)
    while ( start < target_date) :
        start += datetime.timedelta( days = 7)
    start   = start - datetime.timedelta( days = 7)
    # get the first Sunday
    end = start +  datetime.timedelta( days = 7)
    print( start, end)
    # get the data in the week we want
    filter = ( ( dr["formatetime"] > start) & ( dr["formatetime"] < end))
    return  dr[ filter]

################################################################################

# def calculate_kn_distance( gps_data ,k):
#     X = gps_data.as_matrix( columns = [ 'startlat', 'startlng'])
#     kn_distance = []
#     for i in range( len(X)):
#         eucl_dist = []
#         for j in range( len(X)):
#             eucl_dist.append(
#                 math.sqrt(
#                     calc_distance( X[i,0], X[i,1], X[j,0], X[j,1]))
#                     )
#         eucl_dist.sort()
#         kn_distance.append( eucl_dist[k])
#     return kn_distance

################################################################################

#
def discovery_clusters( gps_data, eps_p, ms, labels):
    # encontra os clusters
    coords = gps_data.as_matrix( columns = [ 'startlat', 'startlng'])
    kms_per_radian = 6371.0088
    epsilon = eps_p / kms_per_radian
    db = DBSCAN( eps=epsilon, min_samples=ms, 
            algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
    cluster_labels = db.labels_
    num_clusters = len(set(cluster_labels))
    clusters = pd.Series([coords[cluster_labels == n] for n in range(num_clusters)])
    gps_data[ labels] = cluster_labels
    return clusters



#
def calc_distance( lat1, lon1, lat2, lon2):
    #
    R = 6373.0
    rlat1 = radians( lat1)
    rlon1 = radians( lon1)
    rlat2 = radians( lat2)
    rlon2 = radians( lon2)
    #
    dlon  = rlon2 - rlon1
    dlat  = rlat2 - rlat1
    a = sin(dlat / 2) ** 2 + cos(rlat1) * cos(rlat2) * (sin(dlon / 2) ** 2)
    c = 2 * atan2( sqrt(a), sqrt(1 - a))
    distance = R * c
    return round(distance, 4)



#
def calc_total_distance( df_location, limit_distance):
    #
    distance_total = 0
    count = 0
    #
    for i in range(len(df_location)-1):
        #
        lat1 = df_location.iloc[i]['startlat']
        long1 = df_location.iloc[i]['startlng']
        #
        lat2 = df_location.iloc[i + 1]['startlat']
        long2 = df_location.iloc[i + 1]['startlng']
        #
        distance = calc_distance(lat1, long1, lat2, long2)
        #
        if distance < limit_distance:
            distance_total += distance
            count += 1
        i += 1
    #
    days    =   len( df_location['weekday'].value_counts())
    return distance_total/days, count

# a, b= calc_total_distance( want, 25)



#
def calc_loc_var( gps_data):
    # gps_daily = pd.DataFrame()
    # gps_daily['latitude_var'] = gps_data.groupby('formatetime')['startlat'].var()
    # gps_daily['longitude_var'] = gps_data.groupby(formatetime')['startlng'].var()
    # #divide by zero encountered in log
    # gps_daily['loc_var'] = np.log(gps_daily['latitude_var'] + gps_daily['longitude_var']+0.1)
    # #
    # mean_var = gps_daily['loc_var'].mean()
    # divide by zero encountered in log
    mean_var = np.log( gps_data['startlat'].var() + gps_data['startlng'].var()+0.1)
    return mean_var


#
def calc_max_distance( df_location, limit_distance):
    distance_max = 0
    for i in range(len(df_location) - 1):
        #
        lat1 = df_location.iloc[i]['startlat']
        long1 = df_location.iloc[i]['startlng']
        #
        lat2 = df_location.iloc[i + 1]['startlat']
        long2 = df_location.iloc[i + 1]['startlng']
        #
        distance = calc_distance(lat1, long1, lat2, long2)
        #
        if distance < limit_distance:
            if distance > distance_max:
                distance_max = distance
        i += 1
    return distance_max



#
def calc_max_distance_home(cluster_home, df_location, limit_distance):
    lat_home = cluster_home[0][0]
    lon_home = cluster_home[0][1]
    max_distance = 0
    for i in range(len(df_location) - 1):
        lat1 = df_location.iloc[i]['latitude']
        long1 = df_location.iloc[i]['longitude']
        distance = calc_distance(lat_home,lon_home, lat1, long1)
        if distance < limit_distance:
            if distance > max_distance:
                max_distance = distance
    return max_distance



#
def calc_dis_Nvar( df_location, limit_distance, mean):
    distance_total = 0
    for i in range(len(df_location) - 1):
        #
        lat1 = df_location.iloc[i]['startlat']
        long1 = df_location.iloc[i]['startlng']
        #
        lat2 = df_location.iloc[i + 1]['startlat']
        long2 = df_location.iloc[i + 1]['startlng']
        #
        distance = calc_distance(lat1, long1, lat2, long2)
        #
        if distance < limit_distance:
            distance = (distance - mean) ** 2
            distance_total += distance
        i += 1
    return distance_total


#
def calc_std_loc( gps_data):
    distance,total_dis = calc_total_distance(gps_data, 25)
    mean = distance / total_dis
    dis_Nvar = calc_dis_Nvar(gps_data, 25, mean)
    #
    std = sqrt(dis_Nvar/total_dis)
    #
    return std


#
def get_cluster_home( clusters, gps_data_home):
    idx_home = None
    distance_tmp = 999999
    coords_home = gps_data_home.as_matrix( columns=['startlat', 'startlng'])
    kms_per_radian = 6371.0088
    epsilon = 0.5 / kms_per_radian
    db_home = DBSCAN( eps=epsilon, min_samples=5, algorithm='ball_tree', metric='haversine').fit(np.radians(coords_home))
    cluster_labels_home = db_home.labels_
    num_clusters = len(set(cluster_labels_home))
    cluster_home = pd.Series([coords_home[cluster_labels_home == n] for n in range(num_clusters)])
    # tmp
    lat_home = cluster_home[0][0][0]
    lon_home = cluster_home[0][0][1]
    #print("home: {},{}".format(lat_home,lon_home))
    if  len( clusters) == 1:
        print( 'here')
        return 0
    for i in range( len(clusters)-1):
        lat_cluster = clusters[i][0][0]
        lon_cluster = clusters[i][0][1]
        distance = calc_distance(lat_home,lon_home, lat_cluster, lon_cluster)
        if distance < distance_tmp:
            distance_tmp = distance
            # print(i)
            idx_home = i
    #print("home encontrado: {},{}".format(lat_home,lon_home))
    return idx_home



def calc_total_obs_clusters(clusters):
    total = 0
    for i in range(len(clusters) - 1):
        size = len(clusters[i])
        total += size
    return total



def calc_entropy_loc( clusters):
    entropy = 0
    total_all = calc_total_obs_clusters(clusters)
    for i in range( len(clusters) - 1):
        total_cluster_i = len(clusters[i])
        pi = total_cluster_i/total_all
        entropy += pi * np.log(pi) * -1
    return entropy


def calc_normalized_entropy_loc( clusters):
    entropy = calc_entropy_loc(clusters)
    number_clusters = len(clusters)
    if number_clusters == 1:
        return 0
    normalized_entropy = entropy/np.log(number_clusters)
    return normalized_entropy



def mark_gps_clock( gps_data):
    clock = []
    for row in gps_data['formatetime']:
        h = row.hour
        # get the hours at night
        if  (( h > 0) & ( h < 8)) | ( h > 21) :
            clock.append( 'night')
        else:
            clock.append( 'day')
    gps_data['clock'] = clock
    return




def mark_gps_week( gps_data):
    weekday = []
    for row in gps_data['formatetime']:
        weekday.append( row.weekday())
    gps_data['weekday'] = weekday
    return



def get_cluster_home_night( clusters, gps_data, clusters_night, gps_night):
    if len( clusters_night) == 1:
        if len( clusters_night[0]) == 0:
            return gps_night.cst_label.mode()[0]
    # get cluster of home in night data 
    cst_home    = get_cluster_home( clusters_night, gps_night)
    # get the row name of home cluster
    dfb         = gps_night[ gps_night['cst_label_night']==cst_home].index.values.astype(int)
    # get the cluster in all data
    cluster     = gps_data.loc[ dfb[0], :]['cst_label']
    return cluster



def get_homestay( cst_home, gps_data, days):
    count = 0
    for i in range( len( gps_data)-1):
        if ( ( gps_data['cst_label'].iloc[i] == cst_home) 
            & (gps_data['cst_label'].iloc[i+1] == cst_home)):
            tmp = ( gps_data['formatetime'].iloc[i+1] - gps_data['formatetime'].iloc[i]).total_seconds()
            count += tmp
    ans = count / ( days *24 *60 *60)
    return ans


def get_trans_time( gps_data, days):
    count = 0
    for i in range( len( gps_data)-1):
        if ((gps_data['cst_label'].iloc[i] == -1) 
           & (gps_data['cst_label'].iloc[i+1] == -1)):
            tmp = (gps_data['formatetime'].iloc[i+1] - gps_data['formatetime'].iloc[i]).total_seconds()
            count += tmp
    ans = count / ( days *24 *60 *60)
    return ans



def normalize_coordinate( gps_data):
    #
    lat     = gps_data[ ['startlat']].values
    lat_N   = preprocessing.scale( lat)
    gps_data[ 'startlat_N'] = lat_N
    #
    lng = gps_data[ ['startlng']].values
    lng_N   = preprocessing.scale( lng)
    gps_data[ 'startlng_N'] = lng_N
    return



def get_circadian_movement_home( cst_home, gps_data):
    #
    lat_home     = gps_data[ gps_data[ 'cst_label'] == cst_home][ 'startlat'].values
    lng_home     = gps_data[ gps_data[ 'cst_label'] == cst_home][ 'startlng'].values
    #
    lat_home_mean= np.mean( lat_home)
    lng_home_mean= np.mean( lng_home)
    #
    dis_home = []
    for i in range( len(gps_data)):
        # home
        #if gps_data[ 'cst_label'].iloc[i] == cst_home:
        if False:
            tmp = 0
        # not home
        else:
            a = gps_data[ 'startlat'].iloc[i]
            b = gps_data[ 'startlng'].iloc[i]
            tmp = calc_distance( a, b, lat_home_mean, lng_home_mean)
        dis_home.append( tmp)
    #normalized
    dis_home = preprocessing.scale( dis_home)
    # frequency range of 24 +- 0.5 hrs
    freq = np.linspace(86400-30*60, 86400+30*60, 2*30*60)
    try:
        energy_home  = sum(lombscargle(gps_data['formatetime'], dis_home, freq, normalize=True))
    except ZeroDivisionError:
        return np.nan
    if energy_home > 0:
        return np.log( energy_home )
    else:
        return np.nan




def get_circadian_movement( gps_data, str_lat, str_lng):
    """Calculates the circadian movement based on GPS location for participants
    https://github.com/sosata/CS120DataAnalysis/blob/master/features/estimate_circadian_movement.m
    TODO need to verify the frequency is calculated correctly.
    """
    # frequency range of 24 +- 0.5 hrs
    freq = np.linspace(86400-30*60, 86400+30*60, 2*30*60)
    try:
        energy_lat  = sum(lombscargle(gps_data['formatetime'], gps_data[ str_lat], freq, normalize=True))
        energy_long = sum(lombscargle(gps_data['formatetime'], gps_data[ str_lng], freq, normalize=True))
    except ZeroDivisionError:
        return np.nan
    tot_energy = energy_lat + energy_long
    if tot_energy > 0:
        return np.log( energy_lat + energy_long)
    else:
        return np.nan





###################################################################################

def get_gps_feature( want, days, types):
    eps      =   0.3
    day_type = {
    'base'       : 5,
    'weekday'    : 4,
    'weekend'    : 2,
    }
    min_pts =   days * day_type[ types]
    ### location variance
    loc_var =   calc_loc_var( want)
    ### cluster
    cst     =   discovery_clusters( want, eps, min_pts, 'cst_label')
    cst_num =   len( cst)
    ### get home cluster
    gps_night   = want[ want['clock'] == 'night']
    # get cluster in night data 
    if len( gps_night) == 0:
        gps_night   = want
    cst_night   =   discovery_clusters( gps_night, 
                    eps, min_pts, 'cst_label_night')
    cst_home    =   get_cluster_home_night( cst, want, cst_night, gps_night)
    ### get home stay
    home_stay   =   get_homestay( cst_home, want, days)
    ### get tansition time
    trans_time  =   get_trans_time( want, days)
    ### entropy
    etp     =   calc_entropy_loc( cst)
    ### N_entropy
    etp_N   =   calc_normalized_entropy_loc( cst)
    ### location standard deviation  STD
    #loc_std =   calc_std_loc( want)
    ### total distance
    dis_all, dis_count =   calc_total_distance( want, 25)
    ### get circadian movement
    cir = get_circadian_movement( want, 'startlat', 'startlng')
    ### get circadian movement on normalized coordinates
    normalize_coordinate( want)
    cir_N = get_circadian_movement( want, 'startlat_N', 'startlng_N')
    # get circadian movement on normalized distance from home
    cir_home = get_circadian_movement_home( cst_home, want)
    # Get 10 features
    feature = []
    # 0
    feature.append( etp)
    # 1
    feature.append( etp_N)
    # 2
    feature.append( loc_var)
    # 3
    feature.append( home_stay)
    # 4
    feature.append( trans_time)
    # 5
    feature.append( dis_all)
    # 6
    feature.append( cst_num)
    # 7
    feature.append( cir)
    # 8
    feature.append( cir_N)
    # 9
    feature.append( cir_home)
    return feature



def get_group_data( data, group):
    target  = []
    days    = 0
    ###
    if group == 'weekday':
        target  = data[ data['weekday'] < 5]
        days    = len( target)
        if days == 0:
            target  = data
            days    = len( target)
    ###
    elif group == 'weekend':
        target  = data[ data['weekday'] > 4]
        days    = len( target)
        if days == 0:
            target  = data[ ( data['weekday'] > 2) | ( data['weekday'] < 2)]
            days    = len( target)
    #median
    else:
        target  = data[ data['weekday'] != group ]
        days    = len( target)
    return target, days




###################################################################################
###################################################################################
###################################################################################
###################################################################################


attend = []
attend_len  =   []
###############

# min threshold (includes all thresholds)
gps_threshold = 1000

###############

for f in os.listdir( dir_train):
    if os.path.isdir( dir_train + f):
        # create feature folder
        try:
            if not os.path.exists( dir_train + f + '/' + 'feature'):
                os.makedirs( dir_train + f + '/' + 'feature')
        except OSError:
            print ('Error: Creating directory. ' +  dir_train + f + '/' + 'feature')
        # read file
        try:
            df = dir_train + f + '/' + 'gps_raw.csv'
            dr =  pd.read_csv( df, index_col=False)
            print( len(dr))
            #print( len(dr))
            if len( dr) > gps_threshold:
                attend_len.append( len( dr))
                attend.append( f)
                ##############################################
                print( len(attend))
                print( f)
                ###
                dateFormatter = "%Y-%m-%d %H:%M:%S"
                dr['formatetime'] = pd.to_datetime(dr['formatetime']).dt.strftime( dateFormatter)
                dr['formatetime'] = dr['formatetime'].astype('datetime64[ns]')
                # mark weekday
                mark_gps_week( dr)
                # mark day or night
                mark_gps_clock( dr)
                ###### get groups
                # group 1
                print( 'group 1')
                l = len( dr['weekday'].value_counts())
                base    = get_gps_feature( dr ,l, 'base')
                # group 2
                print( 'group 2')
                data, l = get_group_data( dr, 'weekday')
                weekday = get_gps_feature( data, l, 'weekday')
                # group 3
                print( 'group 3')
                data, l = get_group_data( dr, 'weekend')
                weekend = get_gps_feature( data, l, 'weekend')
                # group 4
                print( 'group 4')
                sub10 = []
                sub10.append( base)
                sub10.append( weekday)
                sub10.append( weekend)
                for i in range( 7):
                    data, l = get_group_data( dr, i)
                    tmp = get_gps_feature( data, l, 'base')
                    sub10.append( tmp)
                median  = np.median( np.array( sub10), axis=0)
                # group 5
                # all group
                all_group   = np.stack( ( base, weekday, weekend, median), axis=0)
                np.savetxt( dir_train + f + '/feature/' + 'gps_'+ str( gps_threshold) +'.csv', all_group, delimiter=",")
        except (FileNotFoundError):
            pass


###################################################################################
###################################################################################
###################################################################################
###################################################################################

### get attend of thresholds

#thresholds = [ 1000, 2000, 4000, 5000, 6000, 7000, 8000, 9000]
thresholds = [ 1000, 2000]
for t in thresholds:
    attend = []
    attend_len  =   []
    ###############
    gps_threshold = t
    print( t)
    ###############
    for f in os.listdir( dir_train):
        if os.path.isdir( dir_train + f):
            # create feature folder
            try:
                if not os.path.exists( dir_train + f + '/' + 'feature'):
                    os.makedirs( dir_train + f + '/' + 'feature')
            except OSError:
                print ('Error: Creating directory. ' +  dir_train + f + '/' + 'feature')
            # read file
            try:
                df = dir_train + f + '/' + 'gps_raw.csv'
                dr =  pd.read_csv( df, index_col=False)
                # print( len(dr))
                ### BREAK POINT
                break_point = -1
                #print( len(dr))
                if len( dr) > gps_threshold:
                    attend_len.append( len( dr))
                    attend.append( f)
            except (FileNotFoundError):
                pass
    ### rerun
    np.savetxt( dir_train + 'gps_attend' + str( gps_threshold) + '.csv', np.array( attend), delimiter=",", fmt="%s")
    len( attend)





###################################################################################



# 蘇泓伊 mhmcph01
# 朱士銓 mhmcph02
# 謝宜廷 mhmcph03
# 徐嘉昊 mhmcph04

# 王怡萱 mhmcmd201
# 林允文 mhmcmd202
# 劉承叡 mhmcmd203
# 廖彥翔 mhmcmd204
# 阮青萍 mhmcmd205

# 王韋凱 mhmcmd101
# 孫上智 mhmcmd102
# 郭旻學 mhmcmd103
# 趙哲宏 mhmcmd104
# 蘇才維 mhmcmd105

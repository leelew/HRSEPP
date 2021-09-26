import calendar
import os

import numpy as np
import requests


#DATA_DIR_L3 = "/hard/lilu/SMAP_L3_NRT/"
DATA_DIR_L4 = "/hard/lilu/SMAP_L4/SMAP_L4/"

USERNAME = 'sysulewlee1@gmail.com'
PASSWORD = '941313Li'


def get_SMAP_L4_path(year, month, day, hour):
    # host url of SMAP
    HOST = 'https://n5eil01u.ecs.nsidc.org'
    # 005 for L4, 004 for L3
    VERSION = '.005'
    # get foldername
    url_path = '{host}/SMAP/SPL4SMGP{version}/{year}.{month:02}.{day:02}/'.\
        format(host=HOST,
               version=VERSION,
               year=year,
               month=month,
               day=day)
    # get filename
    filename = 'SMAP_L4_SM_gph_{year}{month:02}{day:02}T{hour:02}3000_Vv5030_001.h5'.\
        format(year=year,
               month=month,
               day=day,
               hour=hour)
    # get url for SMAP L4
    url_SMAP_L4 = url_path + filename

    return url_SMAP_L4, filename


def get_SMAP_L3_path(year, month, day):
    # host url of SMAP
    HOST = 'https://n5eil01u.ecs.nsidc.org'
    # 005 for L4, 004 for L3
    VERSION = '.007'
    # get foldername
    url_path = '{host}/SMAP/SPL3SMP{version}/{year}.{month:02}.{day:02}/'.\
        format(host=HOST,
               version=VERSION,
               year=year,
               month=month,
               day=day)
    # get filename
    filename = 'SMAP_L3_SM_P_{year}{month:02}{day:02}_R17030_001.h5'.\
        format(year=year,
               month=month,
               day=day)
    # get url for SMAP L3
    url_SMAP_L3 = url_path + filename

    return url_SMAP_L3, filename


def download_SMAP_nrt(year):


    if not os.path.exists(DATA_DIR_L4):
        os.mkdir(DATA_DIR_L4)

    # Use a requests session to keep track of authentication credentials
    with requests.Session() as session:
        session.auth = (USERNAME, PASSWORD)
        for month in np.arange(7, 8):
            _, days_in_month = calendar.monthrange(year, month)

            for day in range(24, days_in_month + 1):
                """
                print('Downloading SMAP L3 data for: '+str(year) +
                      '-'+str(month).zfill(2)+'-'+str(day).zfill(2))

                url_SMAP_L3, file_name_L3 = get_SMAP_L3_path(
                    year, month, day)
                out_path_L3 = os.path.join(DATA_DIR_L3, file_name_L3)
                print(url_SMAP_L3)
                response = session.get(url_SMAP_L3, headers={
                                    'Connection': 'close'})


                # If the response code is 401, we still need to authorize with earthdata.
                if response.status_code == 401:
                    response = session.get(response.url)
                assert response.ok, 'Problem downloading data! Reason: {}'.\
                    format(response.reason)

                with open(out_path_L3, 'wb') as f:
                    f.write(response.content)
                print('*** SMAP L3 data saved to: ' + out_path_L3 + ' *** ')
                """

                folder = DATA_DIR_L4 + '{year}.{month:02}.{day:02}/'.\
                        format(year=year,
                            month=month,
                            day=day)
                if not os.path.exists(folder):
                    os.mkdir(folder)


                for hour in np.arange(1, 24, 3):

                    print('Downloading SMAP L4 data for: ' +
                          str(year) + '-' +
                          str(month).zfill(2)+'-' +
                          str(day).zfill(2)+'-' +
                          str(hour).zfill(2))

                    url_SMAP_L4, file_name_L4 = get_SMAP_L4_path(
                        year, month, day, hour)
                    print(url_SMAP_L4)
                    
                    out_path_L4 = os.path.join(folder, file_name_L4)
                    response = session.get(url_SMAP_L4)

                    # If the response code is 401, we still need to authorize with earthdata.
                    if response.status_code == 401:
                        response = session.get(response.url)
                    assert response.ok, 'Problem downloading data! Reason: {}'.\
                        format(response.reason)

                    with open(out_path_L4, 'wb') as f:
                        f.write(response.content)

                    print('*** SMAP L4 data saved to: ' +
                          out_path_L4 + ' *** ')


if __name__ == '__main__':
    download_SMAP_nrt(year=2016)

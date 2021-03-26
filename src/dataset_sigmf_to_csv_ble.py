from sigmf import SigMFFile, sigmffile
import csv
import numpy as np
import pickle
import sys
import os
import re
import glob
import math
import pandas as pd

#Parses the sigmf file to gather the data into csv files: one global file with data from every scenario called gathered_data_ble.csv, and a file  for each scene


def find_between(s, first, last):
    try:
        start = s.index(first) + len(first)
        end = s.index(last, start)
        return s[start:end]
    except ValueError:
        return ""


def isNaN(string):
    return string != string


dataset_directory = sys.argv[1] + '/ble'
target_dir = sys.argv[2] + '/data/processed'

gathered_name = target_dir + '/gathered_data_ble.csv'

with open(gathered_name, 'w') as csv_write:
    writer_csv = csv.DictWriter(csv_write,
                                fieldnames=[
                                    'Time', 'Len Packet', 'Central Frequency',
                                    'X', 'Y', 'IQ', 'real', 'im', 'Server_id',
                                    'Robot_node', 'Scenario', 'Scene'
                                ])
    writer_csv.writeheader()

    for dirname_scenario in os.listdir(dataset_directory):
        scenario = dirname_scenario[-1]
        path_scenario = dataset_directory + '/' + dirname_scenario
        for dirname_scene in os.listdir(path_scenario):
            scene = dirname_scene[5:]
            path_scene = path_scenario + '/' + dirname_scene
            name = target_dir + '/scenario' + scenario + '_scene' + scene + '.csv'
            with open(name, 'w') as csv_write_scene:
                writer_csv_scene = csv.DictWriter(
                    csv_write_scene,
                    fieldnames=[
                        'Time', 'Len Packet', 'Central Frequency', 'X', 'Y',
                        'IQ', 'real', 'im', 'Server_id', 'Robot_node',
                        'Scenario', 'Scene'
                    ])
                writer_csv_scene.writeheader()
                for dirname_date in os.listdir(path_scene):
                    if dirname_date.startswith('.'):
                        continue
                    for filename in glob.glob(path_scene + '/' + dirname_date +
                                              '/*.sigmf'):
                        print(filename)
                        path_date = path_scene + '/' + dirname_date
                        server = find_between(filename, 'server', '_')
                        robot_node = find_between(filename, 'mobile', '.')
                        if server == '':
                            server = find_between(filename, 'server', '.')

                        signal = sigmffile.fromfile(filename)

                        # Get some metadata and all annotations
                        sample_count = signal.sample_count
                        annotations = signal.get_annotations()

                        # Iterate over annotations
                        for adx, annotation in enumerate(annotations):
                            comment = annotation['core:comment']

                            annotation_start_idx = annotation[
                                'core:sample_start']
                            annotation_length = annotation['core:sample_count']
                            latitude = annotation['core:latitude']
                            longitude = annotation['core:longitude']

                            # Get capture info associated with the start of annotation
                            capture = signal.get_capture_info(
                                annotation_start_idx)
                            freq_center = capture.get('core:frequency', 0)
                            time = capture.get('core:time')

                            if scene == '31':
                                name_extract = path_date + '/extract_' + os.path.splitext(
                                    os.path.basename(filename))[0] + '.csv'
                                df = pd.read_csv(name_extract)
                                robot_node = df.iloc[adx]['Robot_node']

                            # Get the samples corresponding to annotation
                            iq_data = signal.read_samples(
                                annotation_start_idx, annotation_length)

                            end_frame = annotation_start_idx + annotation_length

                            for iq in iq_data:
                                writer_csv.writerow({
                                    'Time': time,
                                    'Len Packet': annotation_length,
                                    'Central Frequency': freq_center,
                                    'X': latitude,
                                    'Y': longitude,
                                    'IQ': iq,
                                    'real': np.real(iq),
                                    'im': np.imag(iq),
                                    'Server_id': server,
                                    'Robot_node': robot_node,
                                    'Scenario': scenario,
                                    'Scene': scene
                                })

                                writer_csv_scene.writerow({
                                    'Time': time,
                                    'Len Packet': annotation_length,
                                    'Central Frequency': freq_center,
                                    'X': latitude,
                                    'Y': longitude,
                                    'IQ': iq,
                                    'real': np.real(iq),
                                    'im': np.imag(iq),
                                    'Server_id': server,
                                    'Robot_node': robot_node,
                                    'Scenario': scenario,
                                    'Scene': scene
                                })

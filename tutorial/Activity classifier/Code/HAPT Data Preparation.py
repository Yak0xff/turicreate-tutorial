# import Turi Create
import turicreate as tc

# define data directory
data_dir = '../HAPT Data Set/RawData/'

# define fine label for containing interval
def find_label_for_containing_interval(intervals, index):
    containing_interval = intervals[:, 0][(intervals[:, 1] <= index) & (index <= intervals[:, 2])]
    if len(containing_interval) == 1:
        return containing_interval[0]

# Load labels
labels = tc.SFrame.read_csv(data_dir + 'labels.txt', delimiter=' ', header=False, verbose=False)
labels = labels.rename({'X1': 'exp_id', 'X2': 'user_id', 'X3': 'activity_id', 'X4': 'start', 'X5': 'end'})

print labels


from glob import  glob

acc_files = glob(data_dir + 'acc_*.txt')
gyro_files = glob(data_dir + 'gyro_*.txt')

# Load data
data = tc.SFrame()
files = zip(sorted(acc_files), sorted(gyro_files))
for acc_file, gyro_file in files:
    exp_id = int(acc_file.split('_')[1][-2:])
    # user_id = int(acc_file.split('_')[2][4:6])

    # Load accel data
    sf = tc.SFrame.read_csv(acc_file, delimiter=' ', header=False, verbose=False)
    sf = sf.rename({'X1': 'acc_x', 'X2': 'acc_y', 'X3': 'acc_z'})
    sf['exp_id'] = exp_id
    # sf['user_id'] = user_id

    # Load gyro data
    gyro_sf = tc.SFrame.read_csv(gyro_file, delimiter=' ', header=False, verbose=False)
    gyro_sf = gyro_sf.rename({'X1': 'gyro_x', 'X2': 'gyro_y', 'X3': 'gyro_z'})
    sf = sf.add_columns(gyro_sf)

    # Calc labels
    exp_labels = labels[labels['exp_id'] == exp_id][['activity_id', 'start', 'end']].to_numpy()
    sf = sf.add_row_number()
    sf['activity_id'] = sf['id'].apply(lambda x: find_label_for_containing_interval(exp_labels, x))
    sf = sf.remove_columns(['id'])

    data = data.append(sf)

print data
print '\n\n\n'

target_map = {
    1.: 'walking',
    2.: 'climbing_upstairs',
    3.: 'climbing_downstairs',
    4.: 'sitting',
    5.: 'standing',
    6.: 'laying'
}

# Use the same labels used in the experiment
data = data.filter_by(target_map.keys(), 'activity_id')
data['activity'] = data['activity_id'].apply(lambda x: target_map[x])
data  = data.remove_column('activity_id')

data.save('hapt_data.sframe')

print data
print '\n\n\n'
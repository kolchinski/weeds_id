import os


CLASS_NAMES = ['chinee_apple',
               'lantana',
               'parkinsonia',
               'parthenium',
               'prickly_acacia',
               'rubber_vine',
               'siam_weed',
               'snake_weed',
               'negatives']


for set_name in ('train', 'val',):
    with open('labels/' + set_name + '_subset0.csv') as f:
        lines = f.readlines()[1:]

    for line in lines:
        filename, label = line.split(',')
        folder_name = CLASS_NAMES[int(label)]
        os.symlink('images/' + filename, 'val/' + folder_name + '/' + filename)

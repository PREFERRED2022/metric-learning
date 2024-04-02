import numpy as np
import torch
from torch.utils.data import Dataset
import random
from tqdm import tqdm


class TripletDataset(Dataset):
    def __init__(self, fire_array, no_fire_array, fire_labels, no_fire_labels, pn_dist_matrix, pp_dist_matrix, np_dist_matrix,
                 bins_dict, pos_bins_dict, neg_bins_dict, pn_ratio=(0, 0, 1), pp_ratio=(0, 0, 1), np_ratio=(0, 1, 0),
                 n_triplets_per_fire_anchor=10, n_triplets_per_no_fire_anchor=1,ratio_fnf=1):
        self.fire_data = fire_array
        self.no_fire_data = no_fire_array
        self.fire_labels = fire_labels
        self.no_fire_labels = no_fire_labels
        self.ratio = pn_ratio
        self.pp_ratio = pp_ratio
        self.n_triplets_per_fire_anchor = n_triplets_per_fire_anchor
        self.n_triplets_per_no_fire_anchor = n_triplets_per_no_fire_anchor
        self.pn_dist_matrix = pn_dist_matrix
        self.pp_dist_matrix = pp_dist_matrix
        self.np_dist_matrix = np_dist_matrix
        self.bins_dict = bins_dict
        self.pos_bins_dict = pos_bins_dict
        self.neg_bins_dict = neg_bins_dict
        self.np_ratio = np_ratio
        self.ratio_fnf = ratio_fnf
        self.triplets = self.generate_triplets_ver3()


    def __getitem__(self, index):
        anchor, positive, negative, mode = self.triplets[index]
        # print(anchor,positive,negative)
        if mode == 'fire':
            anchor_label = 1
            positive_label = 1
            negative_label = 0
            return (self.fire_data[anchor], anchor_label), (self.fire_data[positive], positive_label), (
                self.no_fire_data[negative], negative_label)
        elif mode == 'no-fire':
            anchor_label = 0
            positive_label = 0
            negative_label = 1
            return (self.no_fire_data[anchor], anchor_label), (self.no_fire_data[positive], positive_label), (
                self.fire_data[negative], negative_label)

    def __len__(self):
        return len(self.triplets)

    def calculate_similarities(self):
        return {}

    def get_bin2_indexes(self, bin='bin2'):
        bin2_indexes = []
        for key, bins in self.pn_dist_matrix.items():
            bin2_list = bins.get(bin)
            if bin2_list is not None:
                bin2_indexes.extend(bin2_list)
        return bin2_indexes

    def find_index(self, search_value, bin='bin2'):
        for outer_key, inner_dict in self.pn_dist_matrix.items():
            if search_value in inner_dict[bin]:
                return outer_key
            else:
                continue
        return None

    def generate_triplets(self):
        triplets = []
        for i in range(len(self.fire_data)):
            anchor = i
            anchor_label = self.fire_labels[anchor]
            # pick the number of negative samples for each category
            # pick similar positive samples for each triplet
            negative_samples_distances = self.pn_dist_matrix[str(anchor)]
            positive_distances = self.pp_dist_matrix[str(anchor)]

            n_hard_negatives = int(self.n_triplets_per_fire_anchor * self.ratio[0])
            n_semi_hard_negatives = int(self.n_triplets_per_fire_anchor * self.ratio[1])
            n_easy_negatives = int(self.n_triplets_per_fire_anchor * self.ratio[2])

            #  'very_hard_negatives':'bin5', 'hard_negatives':'bin4',
            # 'semi_hard_negatives':'bin3','easy_negatives':'bin2','very_easy_negatives':'bin1'
            try:
                hard_negatives = np.random.choice(negative_samples_distances[self.bins_dict['hard_negatives']],
                                                  n_hard_negatives)
                n_hard_negatives_selected = len(hard_negatives)
                if n_hard_negatives_selected < n_hard_negatives:
                    extra_needed = n_hard_negatives - n_hard_negatives_selected
                    extra = np.random.choice(negative_samples_distances[self.bins_dict['very_hard_negatives']],
                                             extra_needed)
                    hard_negatives = hard_negatives + extra
            except:
                try:
                    hard_negatives = np.random.choice(negative_samples_distances[self.bins_dict['very_hard_negatives']],
                                                      n_hard_negatives)
                except:
                    continue

            try:
                semi_hard_negatives = np.random.choice(
                    negative_samples_distances[self.bins_dict['semi_hard_negatives']], n_semi_hard_negatives)
                n_semi_hard_negatives_selected = len(semi_hard_negatives)
                if n_semi_hard_negatives_selected < n_semi_hard_negatives:
                    extra_needed = n_semi_hard_negatives - n_semi_hard_negatives_selected
                    extra = np.random.choice(negative_samples_distances[self.bins_dict['easy_negatives']], extra_needed)
                    semi_hard_negatives = semi_hard_negatives + extra
            except:
                try:
                    semi_hard_negatives = np.random.choice(negative_samples_distances[self.bins_dict['easy_negatives']],
                                                           n_semi_hard_negatives)
                except:
                    continue

            try:
                easy_negatives = np.random.choice(negative_samples_distances[self.bins_dict['very_easy_negatives']],
                                                  n_easy_negatives)
                n_easy_negatives_selected = len(easy_negatives)
                if n_easy_negatives_selected < n_easy_negatives:
                    extra_needed = n_easy_negatives - n_easy_negatives_selected
                    extra = np.random.choice(negative_samples_distances[self.bins_dict['easy_negatives']], extra_needed)
                    easy_negatives = easy_negatives + extra
            except:
                # print(i, 'No easy negatives for this fire instance' )
                try:
                    easy_negatives = np.random.choice(negative_samples_distances[self.bins_dict['easy_negatives']],
                                                      n_easy_negatives)
                except:
                    # print(i, 'No semi easy negatives for this fire instance')
                    continue

            # the most similar positive instances are selected for the triplets with fire anchor
            positives = positive_distances[self.pos_bins_dict['very_easy_positives']]
            if positives == []:
                positives = positive_distances[self.pos_bins_dict['easy_positives']]
                if positives == []:
                    positives = positive_distances[self.pos_bins_dict['semi_hard_positives']]

            # if it finds less samples than asked it will take these -> future improv: to take the rest from the previous bin
            for j in range(len(hard_negatives)):
                triplets.append((anchor, np.random.choice(positives), hard_negatives[j], 'fire'))
            for j in range(len(semi_hard_negatives)):
                triplets.append((anchor, np.random.choice(positives), semi_hard_negatives[j], 'fire'))
            for j in range(len(easy_negatives)):
                try:
                    triplets.append((anchor, np.random.choice(positives), easy_negatives[j], 'fire'))
                except:
                    continue
                    #print(anchor,positives, easy_negatives)

        n_fire_anchor_triplets = len(triplets)
        print(f"{n_fire_anchor_triplets} triplets where made for fire anchors")


        # No fire anchor samples

        bin2_indexes = self.get_bin2_indexes(bin = self.bins_dict['easy_negatives'])
        no_fire_data_length = len(self.no_fire_data)
        sample_size = int(len(self.fire_data) * self.ratio_fnf)
        indexes_set = list(set(bin2_indexes))
        len_indexes_set = len(indexes_set)
        if len_indexes_set > sample_size:
            indexes_set = random.sample(indexes_set, sample_size)
            len_indexes_set = len(indexes_set)
        if len_indexes_set < sample_size:
            extra = sample_size - len_indexes_set
            bin3_indexes = self.get_bin2_indexes(bin=self.bins_dict['semi_hard_negatives'])
            extra_indexes_set = list(set(bin3_indexes))
            if extra < len(extra_indexes_set):
                extra_indexes_set = random.sample(extra_indexes_set, extra)
            indexes_set = indexes_set + extra_indexes_set

        del bin2_indexes
        indexes_set = set(indexes_set)
        len_indexes_set = len(indexes_set)

        easy_indexes_dict = {key: [] for key in indexes_set}
        semi_hard_indexes_dict = {key: [] for key in indexes_set}
        hard_indexes_dict = {key: [] for key in indexes_set}

        print('Gathering no fire instances.....')
        for i, no_fire_index in enumerate(indexes_set):
            if self.np_ratio[2] > 0:
                easy_fire_key = self.find_index(no_fire_index, bin=self.bins_dict['easy_negatives'])
                easy_indexes_dict[no_fire_index].append(easy_fire_key)
            if self.np_ratio[1] > 0:
                semi_hard_fire_key = self.find_index(no_fire_index, bin=self.bins_dict['semi_hard_negatives'])
                semi_hard_indexes_dict[no_fire_index].append(semi_hard_fire_key)
            if self.np_ratio[0] > 0:
                hard_fire_key = self.find_index(no_fire_index, bin=self.bins_dict['hard_negatives'])
                hard_indexes_dict[no_fire_index].append(hard_fire_key)

        print('Making triplets with no fire anchors.....')
        for i in tqdm(range(len_indexes_set)):
            n_hard_positives = int(self.n_triplets_per_no_fire_anchor * self.np_ratio[0])
            n_semi_hard_positives = int(self.n_triplets_per_no_fire_anchor * self.np_ratio[1])
            n_easy_positives = int(self.n_triplets_per_no_fire_anchor * self.np_ratio[2])
            for j in range(n_easy_positives):
                item = random.sample(list(easy_indexes_dict.items()), 1)[0]
                key, value = item[0], item[1]
                # del easy_indexes_dict[key]
                triplets.append((key, random.choice(list(indexes_set)), int(value[0]), 'no-fire'))
            for j in range(n_semi_hard_positives):
                item = random.sample(list(semi_hard_indexes_dict.items()), 1)[0]
                key, value = item[0], item[1]
                # del semi_hard_indexes_dict[key]
                try:
                    triplets.append((key, random.choice(list(indexes_set)), int(value[0]), 'no-fire'))
                except:
                    continue
            for j in range(n_hard_positives):
                item = random.sample(list(hard_indexes_dict.items()), 1)[0]
                key, value = item[0], item[1]
                del hard_indexes_dict[key]
                # To change this-->random.choice(list(indexes_set))
                triplets.append((key, int(value[0]), np.random.choice(no_fire_data_length), 'no-fire'))

        n_no_fire_anchor_triplets = len(triplets) - n_fire_anchor_triplets
        print(f"{n_no_fire_anchor_triplets} where made for no fire anchors")
        '''num1 = random.randint(0, 90000)
        file_path = 'triplets_'+str(num1)+'.txt'
        with open(file_path, 'wb') as file:
            # Use pickle.dump() to save the data to the file
            pickle.dump(triplets, file)'''

        return triplets

    def generate_triplets_(self):
        triplets = []
        remainder = 0
        negatives_used = []
        fire_anchor_triplets = 0
        #for i in range(len(self.fire_data)):
        for anchor, anchor_label in tqdm(enumerate(self.fire_labels), total=len(self.fire_data),
                                             desc="Making Fire Triplets"):
            #anchor = i
            #anchor_label = self.fire_labels[anchor]
            # pick the number of negative samples for each category and similar positive samples for each triplet
            negative_samples_distances = self.pn_dist_matrix[str(anchor)]
            positive_distances = self.pp_dist_matrix[str(anchor)]

            n_easy_negatives = int(self.n_triplets_per_fire_anchor * self.ratio[2]) + remainder

            candidates = negative_samples_distances[self.bins_dict['very_easy_negatives']]
            unique_candidates = [neg for neg in candidates if neg not in negatives_used]
            #print('unique_candidates:', unique_candidates)

            easy_negatives = np.random.choice(unique_candidates, min(n_easy_negatives, len(unique_candidates)),
                                              replace=False)

            n_easy_negatives_selected = len(easy_negatives)

            # the most similar positive instances are selected for the triplets with fire anchor
            positives = positive_distances[self.pos_bins_dict['very_easy_positives']]
            if positives == []:
                positives = positive_distances[self.pos_bins_dict['easy_positives']]

            # if it finds less samples than asked it will take these -> future improv: to take the rest from the previous bin
            for j in range(len(easy_negatives)):
                try:
                    triplets.append((anchor, np.random.choice(positives), easy_negatives[j], 'fire'))
                    negatives_used.append(easy_negatives[j])
                except:
                    continue
                    #print(anchor,positives, easy_negatives)

            new_fire_anchor_triplets = len(triplets) - fire_anchor_triplets
            fire_anchor_triplets = len(triplets)
            remainder = n_easy_negatives - new_fire_anchor_triplets
            #print('Remainder:', remainder)

        print(f"{fire_anchor_triplets} triplets where made for fire anchors")


        # No fire anchor samples

        bin2_indexes = self.get_bin2_indexes(bin=self.bins_dict['very_easy_negatives'])
        no_fire_data_length = len(self.no_fire_data)
        sample_size = int(len(self.fire_data) * self.ratio_fnf)
        indexes_set = list(set(bin2_indexes))
        len_indexes_set = len(indexes_set)

        if len_indexes_set > sample_size:
            indexes_set = random.sample(indexes_set, sample_size)
            len_indexes_set = len(indexes_set)

        easy_indexes_dict = {key: [] for key in indexes_set}

        print('Gathering no fire instances.....')
        for i, no_fire_index in enumerate(indexes_set):
            easy_fire_key = self.find_index(no_fire_index, bin=self.bins_dict['easy_negatives'])
            easy_indexes_dict[no_fire_index].append(easy_fire_key)

        print('Making triplets with no fire anchors.....')
        for i in tqdm(range(len_indexes_set)):
            n_easy_positives = int(self.n_triplets_per_no_fire_anchor * self.np_ratio[2])
            for j in range(n_easy_positives):
                item = random.sample(easy_indexes_dict.items(), 1)[0]
                key, value = item[0], item[1]
                # del easy_indexes_dict[key]
                try:
                    triplets.append((key, random.choice(list(indexes_set)), int(value[0]), 'no-fire'))
                except:
                    continue

        n_no_fire_anchor_triplets = len(triplets) - fire_anchor_triplets
        print(f"{n_no_fire_anchor_triplets} where made for no fire anchors")
        '''num1 = random.randint(0, 90000)
        file_path = 'triplets_'+str(num1)+'.txt'
        with open(file_path, 'wb') as file:
            # Use pickle.dump() to save the data to the file
            pickle.dump(triplets, file)'''

        return triplets

    def generate_triplets_ver3(self):
        triplets = []
        for i in range(len(self.fire_data)):
            anchor = i
            anchor_label = self.fire_labels[anchor]
            # pick the number of negative samples for each category
            # pick similar positive samples for each triplet
            negative_samples_distances = self.pn_dist_matrix[str(anchor)]
            positive_distances = self.pp_dist_matrix[str(anchor)]

            n_hard_negatives = int(self.n_triplets_per_fire_anchor * self.ratio[0])
            n_semi_hard_negatives = int(self.n_triplets_per_fire_anchor * self.ratio[1])
            n_easy_negatives = int(self.n_triplets_per_fire_anchor * self.ratio[2])

            #  'very_hard_negatives':'bin5', 'hard_negatives':'bin4',
            # 'semi_hard_negatives':'bin3','easy_negatives':'bin2','very_easy_negatives':'bin1'
            try:
                hard_negatives = np.random.choice(negative_samples_distances[self.bins_dict['hard_negatives']],
                                                  n_hard_negatives)
                n_hard_negatives_selected = len(hard_negatives)
                if n_hard_negatives_selected < n_hard_negatives:
                    extra_needed = n_hard_negatives - n_hard_negatives_selected
                    extra = np.random.choice(negative_samples_distances[self.bins_dict['very_hard_negatives']],
                                             extra_needed)
                    hard_negatives = hard_negatives + extra
            except:
                try:
                    hard_negatives = np.random.choice(negative_samples_distances[self.bins_dict['very_hard_negatives']],
                                                      n_hard_negatives)
                except:
                    continue

            try:
                semi_hard_negatives = np.random.choice(
                    negative_samples_distances[self.bins_dict['semi_hard_negatives']], n_semi_hard_negatives)
                n_semi_hard_negatives_selected = len(semi_hard_negatives)
                if n_semi_hard_negatives_selected < n_semi_hard_negatives:
                    extra_needed = n_semi_hard_negatives - n_semi_hard_negatives_selected
                    extra = np.random.choice(negative_samples_distances[self.bins_dict['easy_negatives']], extra_needed)
                    semi_hard_negatives = semi_hard_negatives + extra
            except:
                try:
                    semi_hard_negatives = np.random.choice(negative_samples_distances[self.bins_dict['easy_negatives']],
                                                           n_semi_hard_negatives)
                except:
                    continue

            try:
                easy_negatives = np.random.choice(negative_samples_distances[self.bins_dict['very_easy_negatives']],
                                                  n_easy_negatives)
                n_easy_negatives_selected = len(easy_negatives)
                if n_easy_negatives_selected < n_easy_negatives:
                    extra_needed = n_easy_negatives - n_easy_negatives_selected
                    extra = np.random.choice(negative_samples_distances[self.bins_dict['easy_negatives']], extra_needed)
                    easy_negatives = easy_negatives + extra
            except:
                # print(i, 'No easy negatives for this fire instance' )
                try:
                    easy_negatives = np.random.choice(negative_samples_distances[self.bins_dict['easy_negatives']],
                                                      n_easy_negatives)
                except:
                    # print(i, 'No semi easy negatives for this fire instance')
                    continue

            # the most similar positive instances are selected for the triplets with fire anchor
            positives = positive_distances[self.pos_bins_dict['very_easy_positives']]
            if positives == []:
                positives = positive_distances[self.pos_bins_dict['easy_positives']]
                if positives == []:
                    positives = positive_distances[self.pos_bins_dict['semi_hard_positives']]

            # if it finds less samples than asked it will take these -> future improv: to take the rest from the previous bin
            for j in range(len(hard_negatives)):
                triplets.append((anchor, np.random.choice(positives), hard_negatives[j], 'fire'))
            for j in range(len(semi_hard_negatives)):
                triplets.append((anchor, np.random.choice(positives), semi_hard_negatives[j], 'fire'))
            for j in range(len(easy_negatives)):
                try:
                    triplets.append((anchor, np.random.choice(positives), easy_negatives[j], 'fire'))
                except:
                    continue
                    #print(anchor,positives, easy_negatives)

        n_fire_anchor_triplets = len(triplets)
        print(f"{n_fire_anchor_triplets} triplets where made for fire anchors")


        # No fire anchor samples

        #bin2_indexes = self.get_bin2_indexes(bin = self.bins_dict['easy_negatives'])
        no_fire_data_length = len(self.no_fire_data)
        sample_size = int(len(self.fire_data) * self.ratio_fnf)

        n_easy_positives = int(sample_size * self.np_ratio[2] * self.ratio_fnf)
        n_semi_hard_positives = int(sample_size * self.np_ratio[1] * self.ratio_fnf)
        n_hard_positives = int(sample_size * self.np_ratio[0] * self.ratio_fnf)

        if n_easy_positives>0:
            bin2_indexes = self.get_bin2_indexes(bin=self.bins_dict['easy_negatives'])
        else:
            bin2_indexes = []
        if n_semi_hard_positives>0:
            bin3_indexes = self.get_bin2_indexes(bin=self.bins_dict['semi_hard_negatives'])
        else:
            bin3_indexes = []
        if n_hard_positives>0:
            bin4_indexes = self.get_bin2_indexes(bin=self.bins_dict['hard_negatives'])
        else:
            bin4_indexes = []
        print('Gathering no fire instances.....')

        i = n_easy_positives
        j = 0
        while i > 0 and j < n_easy_positives: # while we still have triplets to make
            try:
                anchor = random.choice(bin2_indexes)
                positive = random.choice(bin2_indexes)
                negative = random.choice(self.np_dist_matrix[str(anchor)][self.neg_bins_dict['easy_positives']])
                triplets.append((anchor, positive, negative, 'no-fire'))
                i -= 1
                j += 1
            except:
                j += 1
                continue

        i = n_semi_hard_positives
        j = 0
        while i > 0 and j < n_semi_hard_positives: # while we still have triplets to make
            try:
                anchor = random.choice(bin3_indexes)
                positive = random.choice(bin3_indexes)
                negative = random.choice(self.np_dist_matrix[str(anchor)][self.neg_bins_dict['semi_hard_positives']])
                triplets.append((anchor, positive, negative, 'no-fire'))
                i -= 1
                j += 1
            except:
                j += 1
                continue

        i = n_hard_positives
        j = 0
        while i > 0 and j < n_hard_positives: # while we still have triplets to make
            try:
                anchor = random.choice(bin4_indexes)
                positive = random.choice(bin4_indexes)
                negative = random.choice(self.np_dist_matrix[str(anchor)][self.neg_bins_dict['hard_positives']])
                triplets.append((anchor, positive, negative, 'no-fire'))
                i -= 1
                j += 1
            except:
                j += 1
                continue

        n_no_fire_anchor_triplets = len(triplets) - n_fire_anchor_triplets
        print(f"{n_no_fire_anchor_triplets} where made for no fire anchors")

        return triplets

class CustomDatasetClassification(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {
            'data': self.data[idx],
            'label': self.labels[idx]
        }
        return sample

class myDataset(torch.utils.data.Dataset):
    def __init__(self, dataset,mode = 'train'):
        self.mode = mode
#        self.data_path = root
#        self.years = years
#        self.dataset = []
#        for year in self.years:
#            path = self.data_path + '/' + str(year)
#            files = os.listdir(path)
#        self.dataset = triplets.main(mode = 'semi-hard',sim_type='cosine',negative=1,month='august')
        #self.dataset = pd.read_csv('dataset_august_semi-hard_1.csv')
        self.dataset = dataset
        self.num_samples = len(self.dataset)

    def __len__(self):
        return self.num_samples

    def get_sample(self, class_=0, anchor=1):
        class_indices = self.dataset[(self.dataset.loc[:,'fire']==class_)&(self.dataset.index!=anchor)].index
        index = np.random.choice(class_indices,1)
        return index[0]

    def get_record(self,row):
        try:
            record = self.dataset[:,:-1]
        except:
            record = self.dataset.iloc[row].loc[self.dataset.columns != 'fire'].to_numpy()
        try:
            label = self.dataset[-1]
        except:
            label = self.dataset.iloc[row].loc['fire']
        return record,label

    def __getitem__(self, index):
        '''

        :param index: Sample row
        :return: (anchor features, anchor labels), (positive features, positive labels) , (negative features, negative labels)
        '''
        if self.mode == 'train':
            anchor_features, anchor_label = self.get_record(index)
            # Get positive sample
            positive_sample = self.get_sample(class_=anchor_label, anchor=index)
            positive_features, positive_label = self.get_record(row=positive_sample)

            # Get negative sample
            negative_sample = self.get_sample(class_=not anchor_label, anchor=index)
            negative_features, negative_label = self.get_record(row=negative_sample)

            return (torch.from_numpy(anchor_features.astype(float)),torch.tensor(anchor_label).float()), (torch.from_numpy(positive_features.astype(float)),torch.tensor(positive_label).float()), (torch.from_numpy(negative_features.astype(float)),torch.tensor(negative_label).float())
        else:
            anchor_features, anchor_label = self.get_record(index)
            return (torch.from_numpy(anchor_features.astype(float)),torch.tensor(anchor_label).float())

class CustomDatasetClassification(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {
            'data': self.data[idx],
            'label': self.labels[idx]
        }
        return sample
import pandas as pd
import numpy as np
import os
import re
import warnings
import toml
import json
from copy import copy
import sys

# add directory above to path
sys.path.append("..")

# load config
with open("config.toml", "r") as fp:
    config_read = fp.read()
    config_dict = toml.loads(config_read)

DEFAULT_RAND_SEED = config_dict["settings"]["random_seed"]
STIMULI_LABELS_MAP_PATH = config_dict["settings"]["stimuli_labels_path"]

with open(STIMULI_LABELS_MAP_PATH, "r") as fp:
    STIMULI_LABELS_MAP = json.load(fp)
    STIMULI_LABELS_MAP = {
        int(stimulus_id): stimulus_data
        for stimulus_id, stimulus_data in STIMULI_LABELS_MAP.items()
    }


class CaseDatasetReader:
    def __init__(
        self, annotations_dir, physiology_dir, videos=None, subjects=None
    ) -> None:
        self.videos = np.array(videos) or np.arange(1, 9)
        self.subjects = np.array(subjects) or np.arange(1, 31)
        self.subjects_set = set(self.subjects)
        self.data_paths, self.sub_pathsid_map = self.generate_data_paths(
            annotations_dir, physiology_dir
        )
        self.stimuli_labels_map = STIMULI_LABELS_MAP

    def generate_data_paths(self, annotations_dir, physiology_dir):
        "Generate tuples of (subject annotations, subject physiology)"
        paths_list = list()
        path_mapping = dict()
        # iterate over files in annotations and physiology dirs
        for annot_f, physio_f in zip(
            sorted(os.listdir(annotations_dir), key=self.get_subject_num),
            sorted(os.listdir(physiology_dir), key=self.get_subject_num),
        ):
            # ensure that both files are for the same participant
            # if names do not match
            if annot_f != physio_f:
                # ignore if files are readme files
                if ("readme" in annot_f.lower()) and ("readme" in physio_f.lower()):
                    continue
                # raise exception if not readme files (files order do not match)
                raise Exception("Mismatched order")
            subject_id = self.get_subject_num(annot_f)
            if subject_id not in self.subjects_set:
                continue
            # add (annotations_path, physiology_path) for current participant
            paths_list.append(
                (
                    os.path.join(annotations_dir, annot_f),
                    os.path.join(physiology_dir, physio_f),
                )
            )
            path_mapping.setdefault(subject_id, len(paths_list) - 1)
        return paths_list, path_mapping

    @staticmethod
    def get_subject_num(f_name):
        "Get subject (participant) id from file name"
        # search for numbers in file name. Numbers correspond to participants
        res = re.search(r"\d+", f_name)
        # if res is None, then there was no number in f_name
        if res is None:
            return -1
        # if res is not None, extract the number
        return int(res.group())

    def get_subject_data(self, subject_id):
        annotations_path, physiology_path = self.data_paths[
            self.sub_pathsid_map.get(subject_id)
        ]
        return pd.read_csv(annotations_path), pd.read_csv(physiology_path)

    def iterate_subjects_data(self):
        for subject in self.subjects:
            yield (subject, self.get_subject_data(subject))


class CaseDatasetProcessor:
    def __init__(
        self, dataset_reader, replace_videos=True, replace_subjects=True,
    ) -> None:
        "Initialize dataset processor from a reader"
        # videos array
        self.videos = dataset_reader.videos
        # subjects array
        self.subjects = dataset_reader.subjects
        # whether to replace videos
        self.replace_videos = replace_videos
        # whether to replace subjects
        self.replace_subjects = replace_subjects
        # mapping video ids -> new video ids. Anonymization
        self.videos_mapping = self.generate_old_to_new_map(
            original_ids=self.videos, new_ids=None if replace_videos else self.videos
        )
        # mapping subject ids -> new subject ids. Anonymization
        self.subjects_mapping = self.generate_old_to_new_map(
            original_ids=self.subjects,
            new_ids=None if replace_subjects else self.subjects,
        )
        # state dict for storing already processed data - speeds up computations
        self.memory_dict = self._make_memory_dict()

    def _make_memory_dict(self):
        memory_dict = {
            subject_id: {
                video_id: {"train": False, "test": False} for video_id in self.videos
            }
            for subject_id in self.subjects
        }
        return memory_dict

    def generate_old_to_new_map(
        self, original_ids, new_ids=None, id_offset=16, random_seed=DEFAULT_RAND_SEED
    ):
        "Generate new ids for given data. Intended use for videos and subjects."
        if not new_ids:
            # initialize random generator
            rng = np.random.default_rng(seed=random_seed)
            # permute array of new ids - same length as old ids, but random numbers from 0 to 53
            new_ids = rng.permutation(np.arange(id_offset + len(original_ids)))[
                : len(original_ids)
            ]
        # make mapping and return
        old_new_mapping = {int(old_i): int(new_i) for old_i, new_i in zip(original_ids, new_ids)}
        return old_new_mapping

    def compute_intervals(
        self,
        annotations_start_time,
        annotations_end_time,
        times_dict,
        scale_time=1_000,
        annotations_step=5e-2,
        allow_common_train_test_sample=False,
    ):
        "Compute time intervals for splitting the sample into different parts"

        def _times_dict_setup(times_dict):
            "Initialize times dict fields with values from default_times_dict"
            # load default dict
            with open(config_dict["settings"]["default_times_dict_path"], "r") as fp:
                default_times_dict = json.load(fp)
            # update default dict with provided values
            default_times_dict.update(times_dict)
            # return
            return default_times_dict

        if not times_dict:
            raise Exception("Please pass times_dict - a dictionary with time intervals")
        # initialize missing fields, if any
        times_dict = _times_dict_setup(copy(times_dict))
        # compute time length of annotations
        annotations_time = annotations_end_time - annotations_start_time
        # if any of fields in times_dict has negative value
        if negative_times := [
            (time_name, time) for time_name, time in times_dict.items() if time < 0
        ]:
            # make sure it is only 1 field
            assert len(negative_times) <= 1, "Only 1 time interval can be infered"
            negative_time = negative_times[0]  # the only item in the list
            # compute sum of the remaining fields
            positive_times_sum = sum(
                [time for _, time in times_dict.items() if time >= 0]
            )
            # replace negative value with maximum time available
            infered_time = (
                negative_time[0],
                (annotations_time - int(positive_times_sum * scale_time)) / scale_time,
            )
            # make sure that new time has positive value
            # if not, there is nothing left in the dataset to infer
            assert (
                infered_time[1] >= 0
            ), "Required time too large. Time interval is too short to extract wantet time segments."
            # if computed time is smaller than 10s, let the user know
            if infered_time[1] < 10:
                warnings.warn("Infered time <10s - are you sure that it's OK?")
            times_dict[infered_time[0]] = infered_time[1]
        # compute split times - time moments where data is split into different segments
        annotations_step = int(annotations_step * scale_time)
        train_start_time = annotations_start_time + int(
            times_dict["beginning_train_separation"] * scale_time
        )
        train_end_time = train_start_time + int(times_dict["train_time"] * scale_time)
        test_start_time = train_end_time + int(
            times_dict["train_test_separation_time"] * scale_time
        )
        test_annot_start_time = test_start_time + int(
            times_dict["test_ignore_time_front"] * scale_time
        )
        test_annot_end_time = test_annot_start_time + int(
            times_dict["test_time"] * scale_time
        )
        test_end_time = test_annot_end_time + int(
            times_dict["test_ignore_time_end"] * scale_time
        )
        # make sure that train and test data do not have a common part if not specified
        assert (
            train_end_time != test_start_time
        ) or allow_common_train_test_sample, "Train end time == test_start_time"
        # make a dictionary with computed times
        ret_times_dict = {
            "train_start_time": train_start_time,
            "train_end_time": train_end_time,
            "test_start_time": test_start_time,
            "test_annot_start_time": test_annot_start_time,
            "test_annot_end_time": test_annot_end_time,
            "test_end_time": test_end_time,
        }
        # make sure sum of all times is smaller than available time
        assert all(
            map(lambda x: x <= annotations_end_time, ret_times_dict.values())
        ), "Required time too large. Time interval is too short to extract wantet time segments."
        return ret_times_dict

    @staticmethod
    def compute_time_index(wanted_time, time_array, scale_time=1_000):
        wanted_time = wanted_time * scale_time
        if wanted_time < 0:
            wanted_time = time_array.iloc[-1] + wanted_time
        assert (
            wanted_time >= 0
        ), "Not enough time in supplied data. Check provided arguments."
        return time_array[time_array >= wanted_time].index[0]

    @staticmethod
    def process_data(
        annotations, physiology, fix_mismatched=True, cut_physiology=True, inplace=True
    ):
        "Process dataframes so annotations and physiology match, and time columns have correct names"
        if not inplace:
            annotations, physiology = annotations.copy(), physiology.copy()
        if "jstime" in annotations.columns:
            annotations.rename(columns={"jstime": "time"}, inplace=True)
        if "daqtime" in physiology.columns:
            physiology.rename(columns={"daqtime": "time"}, inplace=True)
        # ensure that annotations will always have corresponding physiology
        if fix_mismatched:
            annotations.drop(
                axis="index",
                # get rows of annotations, where time < time at the beginning of physiology and rows of annotations, where time > time at the end of physiology
                index=annotations[
                    (annotations["time"] < physiology["time"].iloc[0])
                    | (annotations["time"] > physiology["time"].iloc[-1])
                ].index,
                inplace=True,
            )
        # make physiology exactly match annotations
        if cut_physiology:
            common_time = (physiology["time"] >= annotations["time"].iloc[0]) & (
                physiology["time"] <= annotations["time"].iloc[-1]
            )
            physiology.drop(
                axis="index",
                # get rows of annotations, where time < time at the beginning of physiology and rows of annotations, where time > time at the end of physiology
                index=physiology[~common_time].index,
                inplace=True,
            )
        start_time = physiology["time"].iloc[0]
        # ensure that time is the first column
        annotations.insert(0, "time", annotations.pop("time") - start_time)
        physiology.insert(0, "time", physiology.pop("time") - start_time)
        # reset time index, so it starts from 0. Drop old index.
        annotations.reset_index(drop=True, inplace=True)
        physiology.reset_index(drop=True, inplace=True)
        # if not inplace processing, return the data
        if not inplace:
            return annotations, physiology

    def extract_video_data(self, subject_data, video, fix_mismatched=True):
        "Get subject data for the specified video"
        # unpack subject's data
        subject_annotations, subject_physiology = subject_data
        # choose data only for the specified video
        video_annotations = subject_annotations[
            subject_annotations["video"] == video
        ].copy()
        video_physiology = subject_physiology[
            subject_physiology["video"] == video
        ].copy()
        # process the data. Inplace by default
        self.process_data(
            video_annotations, video_physiology, fix_mismatched=fix_mismatched
        )
        # return processed data
        return video_annotations, video_physiology

    def extract_data_for_intervals(
        self, annotations, physiology, time_intervals, set_type
    ):
        # times to use
        physiology_start_time, physiology_end_time = (
            time_intervals[f"{set_type}_start_time"],
            time_intervals[f"{set_type}_end_time"],
        )
        # for test use annotation times, not physiology times
        # annotation times may be different
        if set_type == "test":
            annotations_start_time, annotations_end_time = (
                time_intervals[f"{set_type}_annot_start_time"],
                time_intervals[f"{set_type}_annot_end_time"],
            )
        else:
            annotations_start_time, annotations_end_time = (
                physiology_start_time,
                physiology_end_time,
            )
        # annotations extraction
        annotations = annotations[
            (annotations["time"] >= annotations_start_time)
            & (annotations["time"] <= annotations_end_time)
        ]
        # physiology extraction
        physiology = physiology[
            (physiology["time"] >= physiology_start_time)
            & (physiology["time"] <= physiology_end_time)
        ]
        # terurn annotations and physiology
        return annotations, physiology

    def _save_data(
        self,
        data,
        out_dir,
        data_type,
        subject_id,
        video,
        set_type=None,
        fold_idx=None,
        out_file_name=None,
        save_video_num=False,
        reset_time=True,
        reset_time_amount=None,
    ):
        """
        Method for saving processed data
        """

        def _to_csv(data, out_dir, file_name):
            "Save data"
            os.makedirs(out_dir, exist_ok=True)
            data.to_csv(os.path.join(out_dir, file_name), index=False)

        # make copy to prevent ambiguity
        data = data.copy()
        # get new ids (or same - check __init__ args)
        save_subject_id = self.subjects_mapping.get(subject_id)
        save_video_id = self.videos_mapping.get(video)
        # replace video in the dataframe if user wants to save it
        if save_video_num:
            self.replace_video(data)
        # else drop the column
        else:
            data.drop(columns=["video"], inplace=True)
        # if reset_time, reset time index so time starts from 0 or time_offset
        if reset_time:
            data.loc[:, "time"] -= (
                reset_time_amount
                if reset_time_amount is not None
                else data["time"].iloc[0]
            )
        # generate a name for the data file
        out_file_name = (
            out_file_name or f"sub_{save_subject_id}_vid_{save_video_id}.csv"
        )
        # if folds task, make path with folds dir
        if fold_idx is not None:
            out_dir = os.path.join(out_dir, f"fold_{fold_idx}")
        # if specified set {train, test}, make path with the set type
        if set_type is not None:
            out_dir = os.path.join(out_dir, set_type)
        # add info about data type
        out_dir = os.path.join(out_dir, data_type)
        # save data
        _to_csv(data, out_dir, out_file_name)
        # save into in the state dict
        self.memory_dict[subject_id][video][set_type] = True
        # delete the data, so memory is freed faster
        del data

    def replace_video(self, data):
        """Replace video column in the data with new video id.
        For speed this method assigns the same value to the whole dataframe
        i.e. new id of the video in first row"""
        # assign new id to the whole video column
        data.loc[:, "video"] = self.videos_mapping.get(data["video"].iloc[0])

    def save_data(
        self,
        annotations=None,
        out_annotations_dir=None,
        physiology=None,
        out_physiology_dir=None,
        subject_id=None,
        video=None,
        set_type=None,
        fold_idx=None,
        out_file_name=None,
        save_video_num=False,
        reset_time=True,
        reset_time_amount=None,
    ):
        "Method managing saving data for experiments"
        # generate file name
        out_file_name = (
            out_file_name
            or f"sub_{self.subjects_mapping.get(subject_id)}_vid_{self.videos_mapping.get(video)}.csv"
        )
        # save annotations if provided
        if annotations is not None:
            self._save_data(
                data=annotations,
                out_dir=out_annotations_dir,
                data_type='annotations',
                subject_id=subject_id,
                video=video,
                set_type=set_type,
                fold_idx=fold_idx,
                out_file_name=out_file_name,
                save_video_num=save_video_num,
                reset_time=reset_time,
                reset_time_amount=reset_time_amount,
            )
        # save physiology if provided
        if physiology is not None:
            self._save_data(
                data=physiology,
                out_dir=out_physiology_dir,
                data_type='physiological',
                subject_id=subject_id,
                video=video,
                set_type=set_type,
                fold_idx=fold_idx,
                out_file_name=out_file_name,
                save_video_num=save_video_num,
                reset_time=reset_time,
                reset_time_amount=reset_time_amount,
            )

    def save_metadata(self, out_dir):
        "Method for saving subjects and videos maps"
        # save additional info
        if self.replace_videos:
            with open(os.path.join(out_dir, "old_new_videos_map.json"), "w") as fp:
                json.dump(self.videos_mapping, fp, sort_keys=True)
        if self.replace_subjects:
            with open(os.path.join(out_dir, "old_new_subjects_map.json"), "w") as fp:
                json.dump(self.subjects_mapping, fp, sort_keys=True)

    def reset_memory_dict(self):
        "Resets internal memory_dict state dict"
        self.memory_dict = self._make_memory_dict()

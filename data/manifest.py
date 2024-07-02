#!/usr/bin/env python

import json
import multiprocessing
from pathlib import Path
from typing import Dict, List, Union
import sox
import hashlib
import tarfile
import requests
from tqdm.auto import tqdm


class Manifest:
    def __init__(self):
        self.__dict__ = {}

    def __repr__(self):
        return f"<Manifest object at {hex(id(self))}>"

    def __str__(self):
        return self.__repr__()

    def __iter__(self):
        for item in self.__dict__.items():
            yield item

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    @staticmethod
    def _process_utterance(data: Dict) -> Dict:
        """
        A single-thread processing function that operates on a single data item (dict). 
        Expected to be called by multiprocessing pool in `prepare_manifest`. The input
        is of following format:
        {
            'audio_file': name of audio with extension,
            'transcript': transcript as string,
        }
        The output format is:
        {
            utt_id: {
                "transcript": "BLA BLA BLA ...",
                "channels": 1,
                "sample_rate": 16000.0,
                "bitdepth": 16,
                "bitrate": 155000.0,
                "duration": 11.21,
                "num_samples": 179360,
                "encoding": "FLAC",
                "fpath": "/datasets/LibriSpeech/test-clean/5683/32879/5683-32879-0004.flac"
                }
            }
        }
        """
        audio_file, trans = data.values()
        audio_file = Path(audio_file)

        utt_id = Path(audio_file).name
        file_info = sox.file_info.info(str(audio_file))
        file_info["fpath"] = str(audio_file)

        return {
            utt_id: {
                "transcript": trans,
                **file_info,
            }
        }

    @classmethod
    def from_items(cls, data: List[Dict], num_jobs: int = 8):
        """
        Prepare a manifest given a data object of following structure:
        [
            {
                'audio_file': name of audio with extension,
                'transcript': transcript as string,
            }
        ]

        The output format is a dictionary, i.e.:
        {
            utt_id: {
                "transcript": "BLA BLA BLA ...",
                "channels": 1,
                "sample_rate": 16000.0,
                "bitdepth": 16,
                "bitrate": 155000.0,
                "duration": 11.21,
                "num_samples": 179360,
                "encoding": "FLAC",
                "silent": false,
                "fpath": "test-clean/5683/32879/5683-32879-0004.flac"
            },
            ...
        }
        """
        obj = cls()

        # run multiprocessing over all items in data
        with multiprocessing.Pool(num_jobs) as pool:
            dataset = pool.map(cls._process_utterance, [item for item in data])
        dataset = {k: v for item in dataset for k, v in item.items()}
        for k in sorted(dataset):
            obj[k] = dataset[k]

        return obj

    @classmethod
    def load(cls, fpath: Union[str, Path]):
        obj = cls()
        with open(fpath, 'r') as fh:
            if str(fpath).endswith("json"):
                obj = json.load(fh)
            elif str(fpath).endswith("jsonl"):
                obj = [json.loads(line.strip()) for line in fh]
            else:
                ValueError(f"Unexpected manifest extension {fpath}")

        return obj

    def _serialize(self) -> Dict:
        return self.__dict__

    def save(self, fpath: Union[str, Path]) -> None:
        data = self._serialize()
        with open(fpath, "w") as fp:
            if str(fpath).endswith("json"):
                json.dump(data, fp, indent=2)
            elif str(fpath).endswith("jsonl"):
                for item in data.items():
                    fp.write(json.dumps(item) + '\n')
            else:
                ValueError(f"Unexpected manifest extension {fpath}")

    @classmethod
    def _validate(cls, items):
        """
        A single-thread validation routine. Expected to be called by `validate_manifest`
        multiprocessing pool.
        """
        files = []
        for item in items:
            fpath = item["fpath"]
            file_info = sox.file_info.info(str(fpath))

            # Check audio file exists
            assert Path(fpath).is_file(), f"file {fpath} does not exist"

            # Check transcript is not empty
            assert item["transcript"], f"{fpath} transcript is empty"

            # Check relevant audio metadata
            assert (
                item["original_duration"] == file_info["duration"]
            ), f"{fpath} faulty duration"
            assert (
                item["original_num_samples"] == file_info["num_samples"]
            ), f"{fpath} faulty number of samples"

            files.append(fpath)

        # Check there are no duplicate items
        assert len(set(files)) == len(files), "duplicate items in manifest"

    def validate(self, num_jobs: int = 8):
        """
        Validate manifest:
            1) all audio files exist
            2) no transcript is empty
            3) relevant audio metadata is correct
            4) there are no duplicate audio files
        """
        with multiprocessing.Pool(num_jobs) as pool:
            pool.apply_async(self._validate, [item for item in self.__dict__.items()])    

    @property    
    def length(self):
        """The 'value' property getter."""
        return len(self.__dict__)


def download_file(
    url: str, filepath: Union[str, Path], force_download: bool = False
) -> None:
    """
    Download URL to a file. Does not support resume rather creates a
    temp file which is renamed after download finishes.
    """
    filepath = Path(filepath)

    if filepath.is_file():
        if force_download:
            filepath.unlink()
            print(
                f"""
                {filepath} exists but downloading from scratch
                because `force_download` = True.
            """
            )
        else:
            print(
                f"{filepath} exists but skipping download because `force_download` = False."
            )
            return

    temp_filepath = Path(str(filepath) + ".tmp")

    req = requests.get(url, stream=True)
    file_size = int(req.headers["Content-Length"])
    chunk_size = 1024 * 1024  # 1MB
    total_chunks = int(file_size / chunk_size)

    with open(temp_filepath, "wb") as fp:
        content_iterator = req.iter_content(chunk_size=chunk_size)
        for chunk in tqdm(
            content_iterator,
            total=total_chunks,
            unit="MB",
            desc=str(filepath),
            leave=True,
        ):
            fp.write(chunk)

    temp_filepath.rename(filepath)


def md5_checksum(filepath: Union[str, Path], target_hash: str) -> bool:
    """
    Do MD5 checksum.
    """
    filepath = Path(filepath)

    file_hash = hashlib.md5()
    with open(filepath, "rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            file_hash.update(chunk)
    return file_hash.hexdigest() == target_hash


def extract_tar(filepath: Union[str, Path], data_dir: Union[str, Path]) -> None:
    """
    Extract tar files into a folder.
    """
    filepath = Path(filepath)
    data_dir = Path(data_dir)

    if filepath.suffixes == [".tar", ".gz"]:
        mode = "r:gz"
    elif filepath.suffix == ".tar":
        mode = "r:"
    else:
        raise IOError(f"filepath has unknown extension {filepath}")

    with tarfile.open(filepath, mode) as tar:
        members = tar.getmembers()
        for member in tqdm(iterable=members, total=len(members), leave=True):
            tar.extract(path=str(data_dir), member=member)

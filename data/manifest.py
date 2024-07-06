#!/usr/bin/env python

import hashlib
import json
import multiprocessing
import tarfile
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Type, TypeVar, Union

import requests
import sox
from tqdm.auto import tqdm

MANIFEST_TYPE = TypeVar("MANIFEST_TYPE", bound="AudioManifest")


class AudioManifest:
    """
    The format is a dictionary, i.e.:
    {
        utt_id: {
            "transcript": "BLA BLA BLA ...",
            "channels": 1,
            "sample_rate": 16000,
            "bitdepth": 16,
            "bitrate": 155000.0,
            "duration": 11.21,
            "num_samples": 179360,
            "encoding": "FLAC",
            "silent": false,
            "file": "/data/LibriSpeech/test-clean/5683/32879/5683-32879-0004.flac",
            "speaker": "1272-128104",
        },
        ...
    }
    """

    def __init__(self) -> None:
        self.__dict__ = {}

    def __repr__(self) -> str:
        return f"<Manifest object at {hex(id(self))}>"

    def __str__(self) -> str:
        return self.__repr__()

    def __iter__(self):
        for item in self.__dict__.items():
            yield item

    def __getitem__(self, key: str) -> Dict:
        return self.__dict__[key]

    def __setitem__(self, key: str, value: Dict) -> None:
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
            'speaker': speaker_id,
        }
        """
        audio_file, trans, speaker = data.values()
        audio_file = Path(audio_file)

        utt_id = Path(audio_file).name
        file_info = sox.file_info.info(str(audio_file))
        file_info["file"] = str(audio_file)
        file_info["sample_rate"] = int(file_info["sample_rate"])
        file_info["speaker"] = speaker
        del file_info["bitrate"]

        return {
            utt_id: {
                "transcript": trans,
                **file_info,
            }
        }

    @classmethod
    def from_items(
        cls: Type[MANIFEST_TYPE], data: List[Dict], num_jobs: int = 8
    ) -> MANIFEST_TYPE:
        """
        Prepare a manifest given a data object of following structure:
        [
            {
                'audio_file': name of audio with extension,
                'transcript': transcript as string,
                'speaker': speaker_id,
            },
            ...
        ]
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
    def load(cls: Type[MANIFEST_TYPE], fpath: Union[str, Path]) -> MANIFEST_TYPE:
        with open(fpath, "r") as fh:
            if str(fpath).endswith("json"):
                obj = json.load(fh)
            elif str(fpath).endswith("jsonl"):
                obj = cls()
                for line in fh:
                    item = json.loads(line.strip())
                    obj[item[0]] = item[1]
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
                    fp.write(json.dumps(item) + "\n")
            else:
                ValueError(f"Unexpected manifest extension {fpath}")

    @classmethod
    def _validate(cls: Type[MANIFEST_TYPE], items: Dict[str, Dict]) -> None:
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

    def validate(self, num_jobs: int = 8) -> None:
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
    def length(self) -> int:
        """The 'value' property getter."""
        return len(self.__dict__)

    @property
    def ids(self) -> List[str]:
        return list(self.__dict__.keys())

    @property
    def files(self) -> List[str]:
        return [f["file"] for f in self.__dict__.values()]

    def split(self, num_parts: int = None) -> List[MANIFEST_TYPE]:
        pass

    def filter(self, fnc: Callable) -> MANIFEST_TYPE:
        pass

    def merge(self, manifest: Iterable[MANIFEST_TYPE]) -> None:
        pass


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

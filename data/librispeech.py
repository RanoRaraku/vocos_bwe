import argparse
import logging
from pathlib import Path
from typing import Dict, Sequence, Union

from manifest import Manifest, download_file, extract_tar, md5_checksum


def setup_logger():
    logging.basicConfig(
        format="LibriSpeech %(levelname)s: %(asctime)s - %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        level=logging.INFO,
    )


MD5 = {
    "dev-clean": "42e2234ba48799c1f50f24a7926300a1",
    "dev-other": "c8d0bcc9cca99d4f8b62fcc847357931",
    "test-clean": "32fa31d27d2e1cad72775fee3f4849a9",
    "test-other": "fb5a50374b501bb3bac4815ee91d3135",
    "train-clean-100": "2a93770f6d5c6c964bc36631d331a522",
    "train-clean-360": "c0e676e450a7ff2f54aeade5171606fa",
    "train-other-500": "d1a0fd59409feb2c614ce4d30c387708",
}


def get_parser():
    parser = argparse.ArgumentParser(description="LibriSpeech utility parser")
    parser.add_argument(
        "--data_dir",
        default="/home/datasets/LibriSpeech",
        type=str,
        help="Directory to save data and manifests",
    )
    parser.add_argument(
        "--dataset_parts",
        nargs="+",
        default=[
            "dev-clean",
            "dev-other",
            "test-clean",
            "test-other",
            "train-clean-100",
            "train-clean-360",
            "train-other-500",
        ],
        help="Datasets parts to prepare, default=all",
    )
    parser.add_argument(
        "--source_url",
        default="https://www.openslr.org/resources/12/",
        type=str,
        help="Source URL to download dataset from, default=www.openslr.org",
    )
    parser.add_argument(
        "--force_download",
        action="store_true",
        help="Force download in case files exist",
    )
    parser.add_argument(
        "--num_jobs",
        default=8,
        type=int,
        help="Number of parallel jobs manifest preparation default=8",
    )
    parser.add_argument(
        "--skip_prepare_manifests",
        action="store_true",
        help="Skip preparing manifests and only download the dataset",
    )

    return parser


class LibriSpeech:
    def __init__(self, args):
        # super().__init__(Manifest)
        setup_logger()

        self.skip_prepare_manifests = args.skip_prepare_manifests
        self.data_dir = Path(args.data_dir).absolute()
        self.dataset_parts = (
            args.dataset_parts
            if isinstance(args.dataset_parts, list)
            else [args.dataset_parts]
        )
        self.source_url = args.source_url
        self.force_download = args.force_download
        self.num_jobs = args.num_jobs

        self.data_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _get_filestem(fpath: Union[str, Path]) -> Path:
        fpath = Path(fpath).name
        for suffix in Path(fpath).suffixes:
            fpath = fpath.removesuffix(suffix)
        return Path(fpath)

    def download_data(self) -> None:
        """
        Download and untar the dataset.

        :param source_url: str of the URL to download LibriSpeech from
        :param data_dir: str|Path, the path of the dir to storage the dataset.
        :param dataset_parts: "librispeech", "mini_librispeech",
            or a list of splits (e.g. "dev-clean") to download.
        :param force_download: Bool, if True, download the tars no matter if the tars exist.
        """
        data_dir = self.data_dir
        dataset_parts = self.dataset_parts

        # Download
        logging.info("Downloading LibriSpeech")
        for part in dataset_parts:
            url = self.source_url + part + ".tar.gz"
            filepath = data_dir / f"{part}.tar.gz"
            download_file(
                url=url, filepath=filepath, force_download=self.force_download
            )

        # Check MD5
        logging.info("Verifying checksums")
        for part in dataset_parts:
            filepath = data_dir / f"{part}.tar.gz"
            assert md5_checksum(filepath, MD5[part]), f"MD5 checksum failed for {part}"

        # Extract tar files
        logging.info("Extracting *.tar files")

        # Dont create another LibriSpeech subdir by unpacking
        # into `data_dir`/LibriSpeech/LibriSpeech
        if str(data_dir).endswith("LibriSpeech"):
            untar_dir = data_dir.parent

        for part in dataset_parts:
            filepath = data_dir / f"{part}.tar.gz"
            extract_tar(filepath=filepath, data_dir=untar_dir)

        logging.info("Download and extraction successful")

    def parse_trans_file(self, trans_file: Union[str, Path]) -> Dict[str, str]:
        """
        Parse trans.txt into a dictionary where file_id is key and file_path
        is value.
        """
        return {
            line.split()[0]: line.split(maxsplit=1)[1].strip()
            for line in open(trans_file)
        }

    def prepare_manifests(self):
        data_dir: Union[str, Path] = self.data_dir
        dataset_parts: Union[str, Sequence[str]] = self.dataset_parts
        num_jobs: int = self.num_jobs
        audio_ext = "flac"
        trans_ext = "trans.txt"

        for part in dataset_parts:
            logging.info(f"Parsing audio and transcripts files for `{part}`")
            subdir = data_dir / Path(part)
            trans_files = subdir.rglob(f"*.{trans_ext}")
            audio_files = subdir.rglob(f"*[0-9].{audio_ext}")

            trans_dict, audio_dict, speaker_dict = {}, {}, {}
            for f in audio_files:
                fstem = str(self._get_filestem(f))
                audio_dict[fstem] = f.absolute()
                speaker_dict[fstem] = str(f.name).split("-")[0]
            for trans_file in trans_files:
                trans_dict.update(self.parse_trans_file(trans_file))

            # Check we are not missing any transcripts or audios
            valid_ids = set(audio_dict.keys()) & set(trans_dict.keys())
            if len(valid_ids) < len(audio_dict) or len(valid_ids) < len(trans_dict):
                logging.warning(
                    "It appears some (transcript, audio)-pairs are missing. "
                    "Will process only valid pairs"
                )

            input_data = [
                dict(
                    audio_file=audio_dict[valid_id],
                    transcript=trans_dict[valid_id],
                    speaker=speaker_dict[valid_id],
                )
                for valid_id in valid_ids
            ]

            # Create manifest
            logging.info("Generating manifest")
            manifest = Manifest.from_items(input_data, num_jobs)

            logging.info("Validating manifest")
            manifest.validate(num_jobs)

            # Save manifests
            logging.info(
                f"Saving librispeech-{part}.jsonl manifest to disk, "
                f"contains {manifest.length} entries"
            )
            manifest.save(self.data_dir / f"librispeech-{part}.{audio_ext}.jsonl")

    def run(self):
        # Download and extract
        # self.download_data()

        # Prepare and save manifests
        if not self.skip_prepare_manifests:
            self.prepare_manifests()


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    librispeech = LibriSpeech(args)
    librispeech.run()

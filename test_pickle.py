import hashlib
import multiprocessing as mp
import os
import pickle

# from torchdata.dataloader2 import communication
import threading
import time
import argparse
import warnings

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from functools import partial
import sys

from torchdata.dataloader2 import (
    communication,
    DataLoader2,
    MultiProcessingReadingService,
    Prototype2MultiProcessingReadingService,
    PrototypeMultiProcessingReadingService,
)
from torchdata.datapipes.iter import IterableWrapper, IterDataPipe
import pandas as pd
import matplotlib.pyplot as plt


def inc(x):
    return x + 1


def is_odd(x):
    return bool(x % 2)


class PrefetchData:
    def __init__(self, source_datapipe, prefetch):
        self.run_prefetcher = True
        self.prefetch_buffer = []
        self.prefetch = prefetch
        self.source_datapipe = source_datapipe


class PrefetcherIterDataPipe(IterDataPipe):
    def __init__(self, source_datapipe, prefetch=10):
        self.source_datapipe = source_datapipe
        self.prefetch = prefetch
        self.thread = None

    @staticmethod
    def thread_worker(prefetch_data):
        # print(os.getpid(), "!!!!!!!! thread starting")
        # time.sleep(10)
        # print('now creating iterator')
        itr = iter(prefetch_data.source_datapipe)
        # print(os.getpid(), "iterator done")
        stop_iteration = False
        while prefetch_data.run_prefetcher:
            if len(prefetch_data.prefetch_buffer) < prefetch_data.prefetch and not stop_iteration:
                try:
                    # print(os.getpid(), "thread getting item")
                    # if prefetch_data.run_prefetcher:
                    item = next(itr)
                    # print(os.getpid(), "thread getting item complete")
                    prefetch_data.prefetch_buffer.append(item)
                    # print(os.getpid(), "item received and store in buffer of ", len(prefetch_data.prefetch_buffer))
                except (
                    RuntimeError,
                    StopIteration,
                ):  # TODO(VitalyFedyunin): Instead of general exception catch invalid iterator here
                    stop_iteration = True
                except communication.iter.InvalidStateResetRequired:
                    stop_iteration = True
                except communication.iter.TerminateRequired:
                    prefetch_data.run_prefetcher = False
            if stop_iteration and len(prefetch_data.prefetch_buffer) == 0:
                # print(os.getpid(), "all items done, leaving thread myself")
                prefetch_data.run_prefetcher = False
            # print(os.getpid(),'thread wait with full buffer')
            time.sleep(0.00001)
        # print(os.getpid(), "!!!!!!!!  exiting prefetch thread")

    def __iter__(self):
        self.reset()
        # self.run_prefetcher = True
        # print(os.getpid(), ">>>>>>>> start thread")
        prefetch_data = PrefetchData(self.source_datapipe, self.prefetch)
        self.prefetch_data = prefetch_data
        self.thread = threading.Thread(target=PrefetcherIterDataPipe.thread_worker, args=(prefetch_data,), daemon=True)
        self.thread.start()
        i = 0
        while prefetch_data.run_prefetcher:
            if len(prefetch_data.prefetch_buffer) > 0:
                # print(os.getpid(), "main loop returns item from buffer")
                yield prefetch_data.prefetch_buffer[0]
                prefetch_data.prefetch_buffer = prefetch_data.prefetch_buffer[1:]
            else:
                # print('waiting element {}-th time'.format(i))
                # i += 1
                time.sleep(0.00001)
        prefetch_data.run_prefetcher = False
        self.thread.join()
        self.thread = None

    def reset(self):
        # print(os.getpid(), "resetting datapipe")
        if "terminate" in os.environ:
            raise Exception(os.getpid(), "who did it?")
        if self.thread is not None:
            self.prefetch_data.run_prefetcher = False
            self.thread.join()
        # print(os.getpid(), "Reset complete")

    def reset_iterator(self):
        # print(os.getpid(), "reset_iterator called on prefetcher")
        self.reset()


class RangeDebug:
    """
    `__iter__` Creates an iterator of range(x)
    """

    def __init__(self, x):
        self.x = x

    def __iter__(self):
        for i in range(self.x):
            print(os.getpid(), f">>>>>>>> returning {i}")
            yield i


def post_adapter_fn(dp):
    return PrefetcherIterDataPipe(dp, 10)


def map_read(x):
    """
    Read stream and close. Used for tar files.
    """
    data = x[1].read()
    x[1].close()
    return x[0], data


def noop(x):
    return x


def image_loader(path):
    with open(path, "rb") as image:
        data = image.read()
    return data


def transform_md5(x, n_md5):
    long_str = ""
    for i in range(n_md5):  # 8 times use about 10 workers at 100%, need more workers to hit IO bandwidth
        long_str += str(hashlib.md5(x).hexdigest())
    result = hashlib.md5(long_str.encode()).hexdigest()
    size = len(x)
    return "", str(result), size


def get_sample_collate(x):
    return x[0][0]


def map_calculate_md5(x, n_md5):
    """
    Calculate MD5 hash of x[1]. Used by both DataPipes. This is like doing a transform.
    Increasing the number of md5 calculation will determine how much CPU you eat up
    (this is approximate for complexity of transforms).
    Balancing between IO and CPU bound.
    """
    long_str = ""
    for i in range(n_md5):  # 8 times use about 10 workers at 100%, need more workers to hit IO bandwidth
        long_str += str(hashlib.md5(x[1]).hexdigest())
    #  result = hashlib.md5(x[1]).hexdigest()
    result = hashlib.md5(long_str.encode()).hexdigest()
    size = len(x[1])
    return x[0], str(result), size


def check_and_output_speed(prefix: str, function, rs, prefetch=None, dlv1=None, n_md5=None):
    """
    Benchmark the speed of the prefetching setup and prints the results.
    Args:
        prefix: String indicating what is being executed
        function: function that returns a DataPipe
        rs: ReadingService for testing
        prefetch: number of batches to prefetch

    Returns:
        TODO: Return data and charts if possible
    """
    # /dev/nvme1n1p1:
    # Timing cached reads:       33676 MB in  2.00 seconds = 16867.31 MB/sec
    # Timing buffered disk reads: 2020 MB in  3.00 seconds = 673.32 MB/sec
    # [5 workers] tar_dp and MultiProcessingReadingService with prefetch None results are: total time 30 sec, with 200000 items 6533 per/sec, which is 133% of best. 21813 Mbytes with io speed at 712 MBps
    if dlv1 is not None:
        dl = dlv1
        rs_type = "Old DL w/ ImageFolder"
    else:
        dp = function()

        if prefetch is not None:
            dp = PrefetcherIterDataPipe(dp, prefetch)

        # Setup DataLoader, otherwise just use DataPipe
        if rs is not None:
            rs_type = rs.__class__.__name__
            if "Prototype" in rs_type:
                rs_type = "DataLoader2 w/ tar archives"
            else:
                rs_type = "Old DL w/ tar archives"
            dl = DataLoader2(dp, reading_service=rs)
        else:
            dl = dp
            rs_type = "[Pure DataPipe]"

    start = time.time()
    # report = start  # Time since last report, create one every 60s
    items_len = 0  # Number of items processed
    total_size = 0  # Number of bytes processed
    time_to_first = None
    # print('starting iterations')
    for _name, _md5, size in dl:
        if items_len > 10 and time_to_first is None:
            time_to_first = time.time() - start
        total_size += size
        # print(total_size)
        items_len += 1
        # if time.time() - report > 60:  # Create a report very 60s
        # print(f"{items_len} items processed so far {total_size} bytes")
        # report = time.time()

    total = time.time() - start
    speed = int(items_len / total)  # item per sec
    function_name = "ImageFolder" if dlv1 else function.__name__

    io_speed = int(total_size / total / 1024 / 1024)  # size MiBs per sec
    total_size = int(total_size / 1024 / 1024)  # total size in MiBs
    total = int(total)
    print(
        f"{prefix} {function_name} and {rs_type} with prefetch {prefetch} | n_md5 {n_md5} results are: total time {total} sec, with {items_len} items at {speed} files per/sec. {total_size} MiB with io speed at {io_speed} MiBps"
    )
    return prefix, function_name, rs_type, prefetch, total, items_len, speed, total_size, io_speed


def append_result(df, workers, n_tar_files, n_md5, fs, iteration,
                  columns, _prefix, fn_name, rs_type, prefetch, total, items_len, speed, total_size, io_speed):
    return pd.concat(
        [
            df,
            pd.DataFrame(
                data=[[workers, fn_name, rs_type, prefetch, n_md5, total, n_tar_files, items_len, total_size, speed,
                       io_speed, fs, iteration]],
                columns=columns,
            ),
        ]
    )


def save_result(df, csv_name, path=""):

    # Save CSV, you can scp for the file afterwards
    df.to_csv(os.path.join(path, f"{csv_name}.csv"))

    # # Save Plot - we can plot it locally
    # df.set_index("n_workers", inplace=True)
    # df.groupby("RS Type")["io_speed (MB/s)"].plot(legend=True)
    #
    # plt.ylabel("IO Speed (MB/s)")
    # plt.xticks(range(0, max_worker))
    # plt.savefig(os.path.join(path, f"{img_name}.jpg"), dpi=300)


def main(args):

    def tar_dp_n(path, n_items, n_md5):
        tar_files = [f"{path}/images{i}.tar" for i in range(n_items)]
        dp = IterableWrapper(tar_files).shuffle().sharding_filter()
        dp = dp.open_files(mode="b").load_from_tar(mode="r:")
        dp = dp.map(map_read)
        dp = dp.map(partial(map_calculate_md5, n_md5=n_md5))
        # dp = PrefetcherIterDataPipe(dp, PREFETCH_ITEMS)
        return dp

    # TODO: Compare with S3
    def s3_dp(path, n_items):
        pass

    n_tar_files = args.n_tar_files  # Each tar files is ~100MB
    n_prefetch = args.n_prefetch  # 100 by default
    n_md5 = args.n_md5  # 4 by default

    args_fs_str = args.fs.lower()
    if args_fs_str == "local":
        path = "/home/ubuntu"
    elif args_fs_str in ("io2", "gp2", "sc1", "st1", "ssd"):
        path = f"/{args_fs_str}_data"
    elif args_fs_str == "fsx_non_iso":
        path = "/fsx/users/ktse"
    elif args_fs_str in ("ontap", "fsx"):
        path = f"/{args_fs_str}_isolated/ktse"
    else:
        raise RuntimeError(f"Bad args.fs, was given {args.fs}")

    args_file_size_str = args.file_size.lower()
    if args_file_size_str in ("l", "large"):
        path += "/source_data/large_images_tars"
    elif args_file_size_str in ("xl", "xxl"):
        path += f"/source_data/{args_file_size_str}_images_tars"
    else:
        raise RuntimeError(f"Bad args.file_size, was given {args.file_size}")

    print(f"{path = }")

    if args_fs_str == "local":
        image_folder_path = "/home/ubuntu/source_data/image_folder"
    elif args_fs_str == "fsx_non_iso":
        image_folder_path = "/fsx/users/ktse/source_data/image_folder"
    elif args_fs_str in ("ontap", "fsx"):
        image_folder_path = f"/{args_fs_str}_isolated/ktse/source_data/image_folder"
    else:
        image_folder_path = f"/{args_fs_str}_data/source_data/image_folder"

    print(f"{image_folder_path = }")


    columns = ["n_workers", "file_type", "RS Type", "n_prefetch", "n_md5", "total_time", "n_tar_files",
               "n_items", "total_size (MB)", "speed (file/s)", "io_speed (MB/s)", "fs", "iteration"]

    df = pd.DataFrame(columns=columns)

    dp_fn = partial(tar_dp_n, path=path, n_items=n_tar_files, n_md5=n_md5)
    dp_fn.__name__ = "Tar"

    n_runs = args.n_epochs

    for n_workers in [2, 4, 6, 8]:  # range(2, 18, 2):
        for n_prefetch in [n_prefetch]:  # The number of `n_prefetch` doesn't seem to influence the speed

            # Old DataLoader
            # for i in range(1 + n_runs):  # 1 warm-up + n runs
            #     old_rs = MultiProcessingReadingService(num_workers=n_workers, prefetch_factor=n_prefetch)
            #     params = check_and_output_speed(f"[{n_workers} workers]",
            #                                     dp_fn, old_rs, prefetch=n_prefetch, n_md5=n_md5)
            #     df = append_result(df, n_workers, n_tar_files, n_md5, args_fs_str, i, columns, *params)

            # New Prototype RS DataLoader2
            # for i in range(1 + n_runs):  # 1 warm-up + n runs
            #     new_rs = PrototypeMultiProcessingReadingService(num_workers=n_workers, post_adapter_fn=post_adapter_fn)
            #     params = check_and_output_speed(f"[prefetch is True, {n_workers} workers]",
            #                                     dp_fn, new_rs, prefetch=n_prefetch, n_md5=n_md5)
            #
            #     df = append_result(df, n_workers, n_tar_files, n_md5, args_fs_str, i, columns, *params)

            # DLv1 with ImageFolder
            # TODO: Improvement - I can add a function to filter out paths that are not relevant
            n_folders = 1200
            if n_tar_files != n_folders:
                warnings.warn(f"ImageFolder version always read all {n_folders} folder,"
                              f"but n_tar_files is {n_tar_files} != {n_folders}.")
            image_folder = ImageFolder(root=image_folder_path,
                                       transform=partial(transform_md5, n_md5=n_md5), loader=image_loader)
            dlv1 = DataLoader(dataset=image_folder, num_workers=n_workers,
                              prefetch_factor=n_prefetch, collate_fn=get_sample_collate)

            for i in range(1 + n_runs):  # 1 warm-up + n runs
                params = check_and_output_speed(f"[DLv1 ImageFolder {n_workers} workers]",
                                                None, None, prefetch=n_prefetch, dlv1=dlv1, n_md5=n_md5)
                df = append_result(df, n_workers, n_tar_files, n_md5, args_fs_str, i, columns, *params)

    # Save CSV
    print(df)
    save_result(df, csv_name=args.output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fs", type=str,
                        help="FileSystem (e.g. local, io2, gp2, sc1) storing tar files named 'images{i}.tar'",
                        default="local")
    parser.add_argument("--n-epochs", default=3, type=int,
                        help="Number of times to benchmark per setup excluding warm up")
    parser.add_argument("--n-tar-files", default=160, type=int, help="Number of tar files")
    parser.add_argument("--n-prefetch", default=50, type=int, help="Number of batches to prefetch")
    parser.add_argument("--n-md5", default=4, type=int,
                        help="Number of times to compute MD5 hash per file, a proxy for transformation complexity")
    parser.add_argument("--file-size", default="large", type=str,
                        help="image size pixels, large (256x256), XL (512x512), XXL (1024x1024)")
    parser.add_argument("--output-file", default="prefetch_result", type=str,
                        help="output csv file name")
    args = parser.parse_args()
    main(args)

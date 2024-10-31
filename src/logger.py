# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause


import datetime
import logging
import os
import sys
import time
from typing import Literal

import colorlog

LOG_COLOR_CONFIG: dict[str, str] = {
    "DEBUG": "white",
    "INFO": "green",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "bold_red",
}

LOG_LEVEL_CONFIG: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


def set_logger(
    name: str,
    level: str = "DEBUG",
    save_flag: bool = True,
    save_path: str = "./log",
    save_type: Literal["file", "dir"] = "file",
    train_flag: bool = True,
    verbose: bool = False,
) -> logging.Logger:
    """Initialize a logger with specified level, format, and output file options.

    Args:
        name (str): logger name
        level (str, optional): logger level. Choices in [DEBUG, INFO, WARNING, ERROR, CRITICAL]. Defaults to "DEBUG".
        save_flag (bool, optional): whether to save the log to file. Defaults to True.
        save_path (str, optional): the path to save the log. Defaults to None.
        save_type (str, optional): the type of the log file, determine whether to save the log to a file or a directory. Choices in [file, dir]. Defaults to 'file'.
        train_flag (bool, optional): whether the logger saved in name of train or test. Choices in [train, test]. Defaults to True.
        verbose (bool, optional): whether to use verbose format or simple format. Defaults to False.
    Returns:
        logging.Logger: a logger object
    """
    assert save_type in {"file", "dir"}, "save_type must be 'file' or 'dir'"
    if save_path == "./log":
        print('[logger] log `save_path` is not specified, use default path: "./log"')

    # ------ logger ------
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # ------ formatter ------
    console_formatter, file_formatter = get_formatter(verbose)

    # ------ handler ------
    # * sh: stream handler, print log to console | fh: file handler, save log to file
    sh = logging.StreamHandler(stream=sys.stdout)
    # sh.terminator = "\r"
    add_handler(logger, sh, console_formatter, level)
    if save_flag:
        fh = get_file_handler(save_path, save_type, train_flag)
        add_handler(logger, fh, file_formatter, level)

    return logger


def get_formatter(verbose: bool) -> tuple[colorlog.ColoredFormatter, logging.Formatter]:
    """set the formatter of the logger

    Args:
        verbose (bool): use verbose format or simple format.

    Returns:
        list: (console formatter, file formatter)
    """
    level = "%(levelname)s"
    color = "%(log_color)s"
    time = "%(asctime)s"
    msg = "%(message)s"

    if verbose:  # verbose format
        file = "%(filename)s"
        line = "[line:%(lineno)d]"
        console_fmt = f"{color}{time}-{file}-{line}-{level}: {msg}"
        file_fmt = f"{time}-{file}-{line}-{level}: {msg}"
    else:  # simple format
        console_fmt = f"{color}{time}: {msg}"
        file_fmt = f"{time}-{level}: {msg}"
    # * fmt = "%(asctime)s %(name)s %(levelname)s: %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"

    file_formatter = logging.Formatter(fmt=file_fmt, datefmt=date_fmt)
    console_formatter = colorlog.ColoredFormatter(fmt=console_fmt, log_colors=LOG_COLOR_CONFIG, datefmt=date_fmt)
    return console_formatter, file_formatter


def get_file_handler(save_path: str, save_type: str, train_flag: bool) -> logging.FileHandler:
    """
    get the file handler of the logger

    Args:
        save_path (str):
        save_type (str): Choices in [file, dir]
        train_flag (bool): flag of train or test

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        logging.FileHandler: _description_
    """
    if save_path is None:
        raise ValueError("'save_dir' must be specified when 'if_file' is True")

    if save_type == "dir":
        log_file_name = "train_log.log" if train_flag else "test_log.log"
        full_path = os.path.join(save_path, log_file_name)
    elif save_type == "file":
        full_path = save_path.rstrip("/") + ".log"  # rstrip: delete the last char if it is '/'
    else:
        raise ValueError("save_type must be 'file' or 'dir'")
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    mode = "w" if save_type == "dir" else "a"
    return logging.FileHandler(full_path, mode=mode)


def add_handler(logger, handler, formatter, level) -> None:
    """add handler to logger"""
    handler.setLevel(LOG_LEVEL_CONFIG[level])
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def log_example() -> None:
    """logger usage example"""
    ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CODE_PATH = os.path.join(ROOT_PATH, "code")
    LOG_PATH = os.path.join(CODE_PATH, "logs")
    save_time = time.strftime("%Y%m%d-%H%M%S")
    model_name = "model_example"
    dataset_name = "dataset_example"

    save_name = f"{save_time}-{model_name}-{dataset_name}"
    log_path = os.path.join(LOG_PATH, save_name, save_name)
    logger = set_logger(name="exp_log", save_flag=True, save_path=log_path, save_type="file", train_flag=True)

    logger.critical("critical")
    logger.error("error")
    logger.warning("warning")
    logger.info("info")
    logger.debug("debug")


if __name__ == "__main__":
    log_example()

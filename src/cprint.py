# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause


from pprint import pformat, pprint
from typing import Any, Literal, Optional

from pygments import highlight
from pygments.formatters.terminal256 import Terminal256Formatter
from pygments.lexers.python import PythonLexer
from termcolor import colored
from termcolor import cprint as termcolor_cprint

COLOR_MAP = {
    "black": "\033[1;30m",
    "red": "\033[1;31m",
    "green": "\033[1;32m",
    "yellow": "\033[1;33m",
    "blue": "\033[1;34m",
    "purple": "\033[1;35m",
    "cyan": "\033[1;36m",
    "white": "\033[1;37m",
    "reset": "\033[0m",
}


def cprint(*args, **kwargs) -> None:
    """
    Print the given text with color highlighting.

    Args:
        text (Any): The text to be printed.
        color (str, optional): The color to be used for highlighting. Defaults to "yellow". See `COLOR_MAP` for all available colors.
    """
    termcolor_cprint(*args, **kwargs)


def print_color_raw(text: Any, color: str = "yellow") -> None:
    """
    Print the given text with color highlighting.

    Args:
        text (Any): The text to be printed.
        color (str, optional): The color to be used for highlighting. Defaults to "yellow". See `COLOR_MAP` for all available colors.
    """
    text = COLOR_MAP[color] + str(text) + COLOR_MAP["reset"]
    print(text)


def print_color(
    text: Any,
    color: Optional[
        Literal[
            "black",
            "grey",
            "red",
            "green",
            "yellow",
            "blue",
            "magenta",
            "cyan",
            "light_grey",
            "dark_grey",
            "light_red",
            "light_green",
            "light_yellow",
            "light_blue",
            "light_magenta",
            "light_cyan",
            "white",
        ]
    ] = "yellow",
) -> None:
    """
    Print the given text with color highlighting.

    Args:
        text (Any): The text to be printed.
        color (str, optional): The color to be used for highlighting. Defaults to "yellow". See `COLOR_MAP` for all available colors.
    """
    print(colored(text, color))


def pprint_color(text: Any, style: str = "dracula") -> None:
    """
    Pretty-print the given object with color highlighting.

    * pformat: https://docs.python.org/3/library/pprint.html#pprint.pformat
    * PythonLexer: https://pygments.org/docs/lexers/#pygments.lexers.python.PythonLexer
    * Terminal256Formatter: https://pygments.org/docs/formatters/#pygments.formatters.terminal.Terminal256Formatter

    Args:
        obj (Any): The object to be pretty-printed.
        style (str, optional): The style of color highlighting to be used. Defaults to "dracula".
            Available styles: "dracula", "monikai", "one-dark", etc. See `pygments.styles` for all available styles.
    """
    if isinstance(text, str):
        print(highlight(text, PythonLexer(ensurenl=False), Terminal256Formatter(style=style)))
    else:
        print(highlight(pformat(text), PythonLexer(ensurenl=False), Terminal256Formatter(style=style)))


def example() -> None:
    data = [
        {"beta": 0.0003, "exception": None, "pid": 23309, "result": 0.1477420465869912},
        {"beta": 3.0, "exception": None, "pid": 23309, "result": 0.8701576752649646},
        {"beta": 0.0005, "exception": None, "pid": 23310, "result": 0.1477420465869912},
        {
            "beta": 1.0,
            "exception": Exception(
                "You huuu",
            ),
            "pid": 23310,
            "result": None,
        },
        {"beta": 0.001, "exception": None, "pid": 23310, "result": 0.8701576752649646},
        {"beta": 0.5, "exception": None, "pid": 23310, "result": 0.31501170037258275},
        {"beta": 0.003, "exception": None, "pid": 23309, "result": 0.31501170037258275},
        {"beta": 0.3, "exception": None, "pid": 23309, "result": 0.6795055168423021},
        {"beta": 0.005, "exception": None, "pid": 23310, "result": 0.6795055168423021},
        {"beta": 0.1, "exception": None, "pid": 23310, "result": 0.5446195499457295},
        {"beta": 0.01, "exception": None, "pid": 23309, "result": 0.5446195499457295},
        {"beta": 0.05, "exception": None, "pid": 23309, "result": 0.37445485098821485},
        {"beta": 0.03, "exception": None, "pid": 23310, "result": 0.37445485098821485},
    ]

    # === print with color highlighting ===
    print_color(data)
    print_color_raw(data)

    # === print with color highlighting and other attributes ===
    cprint(text="Hello, World!", color="red", on_color="on_black", attrs=["bold", "blink"])

    # === print with pretty format ===
    pprint(object=data)
    pprint_color(text=data)


if __name__ == "__main__":
    example()

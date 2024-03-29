"""
Python Wechaty - https://github.com/wechaty/python-wechaty-puppet

Authors:    Huan LI (李卓桓) <https://github.com/huan>
            Jingjing WU (吴京京) <https://github.com/wj-Mcat>

2018-now @copyright Wechaty

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from __future__ import annotations

import os


def get_or_create_dir(*paths) -> str:
    """
    get or create path
    Args:
        paths: the sub path of
    Returns:
    """

    path = os.path.join(*paths)
    os.makedirs(path, exist_ok=True)
    return path


# global cache dir
CACHE_DIR: str = get_or_create_dir('.wechaty')

"""
docstring
"""
from __future__ import annotations

import json
import base64
import requests
import os
import base64
from collections import defaultdict
import mimetypes

from typing import (
    Type,
    Optional,
    Union,
)

import qrcode   # type: ignore

from .utils import extract_file_name_from_url, data_url_to_base64, get_json_data

from .type import (
    FileBoxOptionsFile,
    FileBoxOptionsUrl,
    FileBoxOptionsStream,
    FileBoxOptionsBuffer,
    FileBoxOptionsQrCode,
    FileBoxOptionsBase64,
    FileBoxOptionsBase,
    Metadata, FileBoxType)

from wechaty_puppet.exceptions import WechatyPuppetConfigurationError
from wechaty_puppet.logger import get_logger


logger = get_logger("FileBox")


class FileBoxEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, bytes):
            return str(obj, encoding='utf-8')

        return json.JSONEncoder.default(self, obj)


class FileBox:
    """
    # TODO -> need to implement pipeable
    maintain the file content, which is sended by wechat
    """

    def __init__(self, options: FileBoxOptionsBase):
        logger.info('init file-box<%s, %s>',  type(options), options.name or '')

        # TODO: will be deprecated after: Dec, 2022
        self._mimeType: Optional[str] = None

        self.mediaType: Optional[str] = None

        self.metadata: Metadata = defaultdict()

        self.name = options.name

        # TODO: will be deprecated after: Dec, 2022
        self.boxType: int = options.type.value
        self._type: int = self.boxType   # type: ignore

        if isinstance(options, FileBoxOptionsFile):
            self.localPath = options.path

        elif isinstance(options, FileBoxOptionsBuffer):
            self.buffer = options.buffer

        elif isinstance(options, FileBoxOptionsUrl):
            self.remoteUrl = options.url
            self.headers = options.headers

        elif isinstance(options, FileBoxOptionsStream):
            # TODO -> need to get into detail for stream sending
            self.stream = options.stream

        elif isinstance(options, FileBoxOptionsQrCode):
            self.qrCode = options.qr_code

        elif isinstance(options, FileBoxOptionsBase64):
            self.base64: bytes
            if isinstance(options.base64, str):
                self.base64 = str.encode(options.base64)
            elif isinstance(options.base64, bytes):
                self.base64 = options.base64
            else:
                raise WechatyPuppetConfigurationError(
                    f'Base64 File Data Type is invalid, str/bytes is supported')
    @property
    def mimeType(self) -> Optional[str]:
        logger.warn(
            'mimeType will be deprecated after Dec, 2022, '
            'we suggest that you should use mediaType property'
        )
        return self.mediaType
    
    @mimeType.setter
    def mimeType(self, value: str):
        logger.warn(
            'mimeType will be deprecated after Dec, 2022, '
            'we suggest that you should use mediaType property'
        )
        self.mediaType = value

    def type(self) -> FileBoxType:
        """get filebox type"""
        return FileBoxType(self._type)
    
    async def ready(self):
        """
        sync the name from remote
        """
        if self.type() == FileBoxType.Url:
            await self.sync_remote_name()

    async def sync_remote_name(self):
        """sync remote name
        refer : https://developer.mozilla.org/en-US/docs/Web/HTTP/
                Headers/Content-Disposition

                Content-Disposition: attachment; filename="cool.html"

        # wujingjing comment 2020-6-29:  according the implementation of
            file-box: https://github.com/huan/file-box/blob/e42d7207bb1cf5
                b76afb8ead6f72715f4a197b35/src/misc.ts#L66

            headers in requests package doesn't contains attribute:
                content-disposition, so we need to change the way to sync
                file-name from url type

            I find the better way to extract file name from url:
                https://stackoverflow.com/questions/18727347/how-to-extract-a-
                filename-from-a-url-append-a-word-to-it/18727481#18727481
        """
        file_box_type = self.type()
        if file_box_type != FileBoxType.Url:
            raise TypeError('type <{0}> is not remote'.format(
                file_box_type.name))

        if not hasattr(self, 'remoteUrl'):
            raise AttributeError('not have attribute url')

        url = getattr(self, 'remoteUrl')
        self.name, self.mimeType = extract_file_name_from_url(url)

    def to_json_str(self) -> str:
        """
        dump the file content to json object
        :return:
        """
        # 1. if sending voice file, so check the metadata
        if self.name.endswith('.sil'):
            if not self.metadata or not self.metadata.get('voiceLength', None):
                logger.warn(
                    'detect that you want to send voice file, '
                    'but metadata is not valid, please set it, '
                    'eg: file_box.metadata = {"voiceLength": 2000}'
                )

        json_data = get_json_data(self)

        # make type in the serialized json data
        if 'boxType' not in json_data:
            json_data['boxType'] = self.type().value

        data = json.dumps(json_data, cls=FileBoxEncoder, indent=4)
        return data

    async def to_file(self, file_path: Optional[str] = None,
                      overwrite: bool = False):
        """
        save the content to the file
        :return:
        """
        file_box_type = self.type()
        if file_box_type == FileBoxType.Url:
            if not self.mimeType or not self.name:
                await self.sync_remote_name()

        file_path = self.name if file_path is None else file_path

        if os.path.exists(file_path) and not overwrite:
            raise FileExistsError(f'FileBox.toFile(${file_path}): file exist. '
                                  f'use FileBox.toFile(${file_path}, true) to '
                                  f'force overwrite.')

        if file_box_type == FileBoxType.Buffer:
            with open(file_path, 'wb+') as f:
                f.write(self.buffer)

        elif file_box_type == FileBoxType.Url:
            with open(file_path, 'wb+') as text_io:
                # get the content of the file from url
                res = requests.get(self.remoteUrl)
                text_io.write(res.content)

        elif file_box_type == FileBoxType.QRCode:
            with open(file_path, 'wb+') as f:
                # create the qr_code image file
                img = qrcode.make(self.qrCode)
                img.get_image().save(f)

        elif file_box_type == FileBoxType.Base64:
            data = base64.b64decode(self.base64)
            with open(file_path, 'wb') as f:
                f.write(data)

        elif file_box_type == FileBoxType.Stream:
            with open(file_path, 'wb+') as f:
                f.write(self.stream)

    def to_base64(self) -> str:
        """
        transfer file-box to base64 string
        :return:
        """
        # TODO -> need to implement other data format
        return ''

    @classmethod
    def from_url(cls: Type[FileBox], url: str, name: Optional[str],
                 headers: Optional[dict] = None) -> FileBox:
        """
        create file-box from url
        """
        if name is None:
            response = requests.get(url)
            # TODO -> should get the name of the file
            name = response.content.title().decode(encoding='utf-8')
        options = FileBoxOptionsUrl(name=name, url=url, headers=headers)
        file_box: FileBox = cls(options)
        file_box.mimeType = mimetypes.guess_type(url)[0] or ''
        return file_box

    @classmethod
    def from_file(cls: Type[FileBox], path: str, name: Optional[str] = None) -> FileBox:
        """create FileBox from local file, which can be any type of file

        Examples:
            >>> file_box = FileBox.from_file('beautiful-gril.png')
            >>> # or
            >>> file_box = FileBox.from_file('hello.sil')
            >>> file_box.metadata = {'voiceLength': 2000}

        Returns:
            _description_
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f'{path} file not found')
        
        if name is None:
            name = os.path.basename(path)
        
        # 1. check the audio related code
        if name.endswith('.silk') or name.endswith('.slk'):
            logger.warn('detect that you want to send voice file which should be <name>.sil pattern. So we help you rename it.')
            if name.endswith('.silk'):
                name = name.replace('.silk', '.sil')
            if name.endswith('.slk'):
                name = name.replace('.slk', '.sil')

        with open(path, 'rb') as f:
            content = base64.b64encode(f.read())

        file_box: FileBox = cls.from_base64(base64=content, name=name)

        # if sending the voice file, the mediaType must be: 'audio/silk'
        if file_box.name.endswith('.sil'):
            file_box.mediaType = 'audio/silk'
            
            if not file_box.metadata or not file_box.metadata.get('voiceLength', None):
                logger.warn(
                    'detect that you want to send voice file, but no voiceLength setting, '
                    'so use the defualt settign: file_box.metadata = {"voiceLength": 1000}'
                    'you should set it manually'
                )
                file_box.metadata = {
                    "voiceLength": 1000
                }
        else:
            file_box.mimeType = mimetypes.guess_type(path)[0] or ''

        return file_box

    @classmethod
    def from_stream(cls: Type[FileBox], stream: bytes, name: str) -> FileBox:
        """
        create file-box from stream

        TODO -> need to implement stream detials
        """
        options = FileBoxOptionsStream(name=name, stream=stream)
        return cls(options)

    @classmethod
    def from_buffer(cls: Type[FileBox], buffer: bytes, name: str) -> FileBox:
        """
        create file-box from buffer

        TODO -> need to implement buffer detials
        """
        options = FileBoxOptionsBuffer(name=name, buffer=buffer)
        return cls(options)

    @classmethod
    def from_base64(cls: Type[FileBox], base64: bytes, name: str = 'base64.dat') -> FileBox:
        """
        create file-box from base64 str

        refer to the file-box implementation, name field is required.

        :param base64:
            example data: data:image/png;base64,${base64Text}
        :param name: name the file name of the base64 data
        :return:
        """
        options = FileBoxOptionsBase64(base64=base64, name=name)
        return FileBox(options)

    @classmethod
    def from_data_url(cls: Type[FileBox], data_url: str, name: str) -> FileBox:
        """
        example value: dataURL: `data:image/png;base64,${base64Text}`,
        """
        return cls.from_base64(
            str.encode(data_url_to_base64(data_url)),
            name
        )

    @classmethod
    def from_qr_code(cls: Type[FileBox], qr_code: str) -> FileBox:
        """
        create file-box from base64 str
        """
        options = FileBoxOptionsQrCode(name='qrcode.png', qr_code=qr_code)
        return cls(options)

    @classmethod
    def from_json(cls: Type[FileBox], obj: Union[str, dict]) -> FileBox:
        """
        create file-box from json data

        TODO -> need to translate :
            https://github.com/huan/file-box/blob/master/src/file-box.ts#L175

        :param obj:
        :return:
        """
        if isinstance(obj, str):
            json_obj = json.loads(obj)
        else:
            json_obj = obj

        # original box_type field name is boxType
        if 'boxType' not in json_obj:
            raise Exception('box field must be required')
        # assert that boxType value must match the value of FileBoxType values

        if json_obj['boxType'] == FileBoxType.Base64.value:
            file_box = FileBox.from_base64(
                base64=json_obj['base64'],
                name=json_obj['name']
            )
        elif json_obj['boxType'] == FileBoxType.Url.value:
            file_box = FileBox.from_url(
                url=json_obj['remoteUrl'],
                name=json_obj['name']
            )
        elif json_obj['boxType'] == FileBoxType.QRCode.value:
            file_box = FileBox.from_qr_code(
                qr_code=json_obj['qrCode']
            )
        else:
            raise ValueError('unknown file_box json object %s',
                             json.dumps(json_obj))

        if 'metadata' not in json_obj:
            json_obj['metadata'] = {}
        elif not isinstance(json_obj['metadata'], dict):
            raise AttributeError('metadata field is not dict type')

        file_box.metadata = json_obj['metadata']

        return file_box

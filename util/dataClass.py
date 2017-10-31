from abc import ABC, abstractmethod
from pathlib import Path

class opts:
    inputPath=None
    outputPath=None

    def __init__(self, inputPath="", outputPath=""):
        self.inputPath=Path(inputPath)
        self.outputPath=Path(outputPath)

class BaseDataClass(ABC):

    def __init__(self, data='hi', opts=[]):
        self.data=self._loadData(data)
        self.opts=opts

    @abstractmethod
    def _loadData(self,data):
        return NotImplemented

class formatHTMLDownload(BaseDataClass):

    def _loadData(self,data):
        return data

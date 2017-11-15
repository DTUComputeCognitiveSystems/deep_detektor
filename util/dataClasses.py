from .baseClasses import BaseDataClass
from pathlib import Path
import pickle

# ==============================================================================
# General data types
# ==============================================================================

class DataPath(BaseDataClass):

    def name(self):
        return 'dataPath'

    def save(self):
        pass

    def load(self):
        pass

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self,data):
        self._data = Path(data);


class DictData(BaseDataClass):

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self,data):
        self._data = data;

    def save(self):
        with open(str(self.opts.outputPath)+'/'+self.name()+'.pickle', 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(self.data, f)

    def load(self):
        with open(str(self.opts.outputPath)+'/'+self.name()+'.pickle', 'rb') as f:
            self.data = pickle.load(f)

    def saveDictToFiles(self):
        for key, value in self.data.items():
            savePath=self.opts.outputPath.joinpath('program'+str(key))
            savePath.write_text(value)
        self.logger.info('All files written to local')

class BaseTable(BaseDataClass):

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self,data):
        self._data = data;

    def save(self):
        self.logger.warning('Not implemented')

    def load(self):
        self.logger.warning('Not implemented')

class CombinedData(BaseDataClass):

    def save(self):
        for dataElement in self.data:
            dataElement.save()

    def load(self):
        self.logger.warning('Not implemented')

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data=data;

    @property
    def name(self):
        return "CombinedData"

# ==============================================================================
# Specialized data
# ==============================================================================

class DICTPrograms(DictData):

    def name(self):
        return 'DICTPrograms'

class HTMLInput(DictData):

    def name(self):
        return 'HTMLInput'

class TABLEData(BaseTable):

    def name(self):
        return 'TABLEData'

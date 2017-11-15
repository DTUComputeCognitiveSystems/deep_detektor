from abc import ABC, abstractmethod
import logging
from pathlib import Path

class opts:

    def __init__(self, inputPath='', outputPath=''):
        self.inputPath=Path(inputPath)
        self.outputPath=Path(outputPath)

class BaseDataClass(ABC):

    def __init__(self, data=None,opts=[]):
        self.logger=logging.Logger(self.name)
        ch = logging.StreamHandler()
        self.logger.addHandler(ch)
        self.logLevel = logging.INFO

        self.data=data
        self.opts=opts

    @abstractmethod
    def save(self):
        return NotImplemented

    @abstractmethod
    def load(self):
        return NotImplemented

    @property
    def data(self):
        return self._data

    @data.setter
    @abstractmethod
    def data(self, data):
        return NotImplemented

    @property
    @abstractmethod
    def name(self):
        return NotImplemented

    @property
    def logLevel(self):
        return self._logLevel

    @logLevel.setter
    def logLevel(self, logLevel):
        self._logLevel=logLevel
        self.logger.handlers[0].setLevel(logLevel)


# ==============================================================================
# Abstract Composition Class
# ==============================================================================

class BaseDataTransformer(ABC):
    """
    Interface for all data transformations.

    The implementation is based on the composition design pattern with
    root nodes being concrete transformations and parents in form of pipelines
    """

    def __init__(self,inputFormat,outputFormat):
        self.parent = None
        self.inputFormat=inputFormat
        self.outputFormat=outputFormat

        self.logger=logging.Logger('dataTransformer')
        ch = logging.StreamHandler()
        self.logger.addHandler(ch)
        self.logLevel = logging.WARNING

    def transform(self, dataInput):
        dataOutput = self._transform(dataInput)
        dataOutput.opts=dataInput.opts
        return dataOutput

    @abstractmethod
    def _transform(self, data):
        pass

#    @property
#    @abstractmethod
#    def name(self):
#        return NotImplemented

    @property
    def inputFormat(self):
        return self._inputFormat

    @inputFormat.setter
    def inputFormat(self, inputFormat):
        self._inputFormat = inputFormat

    @property
    def outputFormat(self):
        return self._outputFormat

    @outputFormat.setter
    def outputFormat(self, outputFormat):
        self._outputFormat = outputFormat

    @property
    def logLevel(self):
        return self._logLevel

    @logLevel.setter
    def logLevel(self, logLevel):
        self._logLevel=logLevel
        self.logger.handlers[0].setLevel(logLevel)

    def listChildren(self):
        pass

    def listParents(self):
        if self.parent is not None:
            print(self.parent)
            if self.parent.parent is not None:
                self.parent.listParents()
        else:
            print("This is the Root")


# ==============================================================================
# Concrete General Composition Elements
# ==============================================================================

class BasePipeline(BaseDataTransformer):
    """
    Pipeline of several data transformations.

    A pipeline may consist of both data transformers, predictive models, and
    pipelines or any combination of these
    """

    def __init__(self, dataTransformers):
        self.dataTransformers = []
        for dataTransformer in dataTransformers:
            self.addChild(dataTransformer)

    def _transform(self,data):
        for dataTransformer in self.dataTransformers:
            data = dataTransformer.transform(data)
        return data


    def addChild(self, dataTransformer):
        # First element
        if not self.dataTransformers:
            self.dataTransformers.append(dataTransformer)
            self.inputFormat = dataTransformer.inputFormat
            self.outputFormat = dataTransformer.outputFormat
        else:
            if (self.outputFormat != dataTransformer.inputFormat):
                print("Error: Output format does not fit to input format")
                print(self.outputFormat)
                print(dataTransformer.inputFormat)
            else:
                dataTransformer.parent = self
                self.dataTransformers.append(dataTransformer)
                self.outputFormat = dataTransformer.outputFormat


        return self

    def listChildren(self):
        for dataTransformer in self.dataTransformers:
            print(dataTransformer)

    @property
    def logLevel(self):
        return self._logLevel

    @logLevel.setter
    def logLevel(self, logLevel):
        for dataTransformer in self.dataTransformers:
            dataTransformer.logLevel=logLevel


class BaseAggregator(BaseDataTransformer):

    def __init__(dataTransformers):
        self.dataTransformers = dataTransformers

    def _transform(self,data):
        for i, dataObject in enumerate(data):
            for dataTransformer in self.dataTransformers:
                if dataTransformer.startswith(dataObject.Name):
                    dataObject=dataTransformer.transform(dataObject)
            data[i]=dataObject
        return data

    @property
    def name(self):
        return "DataAggregator"

class BasePredictiveModel(BaseDataTransformer):
    """
    Predictive models are pre-trained models taking tensor input and outputting
    corresponding predictions

    The class need to be given such a model in order to be constructed.
    """

    def __init__(self,model):
        self.model = model
        super().__init__('X','yPredictions')

    def _transform(self,data):
        data = self.model.predict(data)
        return data

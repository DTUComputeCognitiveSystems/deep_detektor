from abc import ABC, abstractmethod

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

    @abstractmethod
    def transform(self, data):
        pass

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

    def transform(self,data):
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

class BasePredictiveModel(BaseDataTransformer):
    """
    Predictive models are pre-trained models taking tensor input and outputting
    corresponding predictions

    The class need to be given such a model in order to be constructed.
    """

    def __init__(self,model):
        self.model = model
        super().__init__('X','yPredictions')

    def transform(self,data):
        data = self.model.predict(data)
        return data

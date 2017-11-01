from .baseClasses import BasePipeline
from .dataTransformers import *

# ==============================================================================
# Concrete Pipelines Implementations
# ==============================================================================

class HTMLSource2HTMLInput(BasePipeline):

    def __init__(self):
        dataTransformers = [fromHTMLSourceToJSONPrograms(),
                            fromJSONProgramsToHTMLInput()]
        # Setup the pipeline
        super().__init__(dataTransformers)

from pathlib import Path

class Paths():
    inputPath
    outputPath

    def __init__(self, inputPath=[], outputPath=[]):
        self.inputPath = Path(inputPath)
        self.outputPath = Path(outputPath)

from .base import Operation, OperationOnFile, OperationOnSubdirectories, OpeartionWithShellCall, RoutineOnDirectory
from jfs.api import Path,Directory,File,mkdir,mv,cp,rm
from typing import TypeVar, Iterable
from pygate.components.simulation import Simulation
from pygate.scripts.shell import Script
import rx


class KEYS:
    SUBDIRECTORIES = 'subdirectories'
    TARGET = 'target'
    IS_TO_BROADCAST = 'is_to_broadcast'
    CONTENT = 'content'
    TO_BROADCAST_FILES = 'to_broadcast_files'


class OpSubdirectoriesMaker(Operation):
    def __init__(self, nb_split: int, subdirectory_format: str="sub.{}"):
        self.nb_split = nb_split
        self.fmt = subdirectory_format

    def apply(self, r: RoutineOnDirectory):
        result = self.dryrun(r)
        for n in result[KEYS.SUBDIRECTORIES]:
            r.directory.makedir(n)
        return result

    def dryrun(self, r: RoutineOnDirectory):
        return {KEYS.SUBDIRECTORIES: tuple([self.fmt.format(i) for i in range(self.nb_split)])}
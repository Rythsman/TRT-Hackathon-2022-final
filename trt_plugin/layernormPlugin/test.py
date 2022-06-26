import os
import sys
import tensorrt as trt
import ctypes
from glob import glob

planFilePath = './'
soFileList = glob(planFilePath + "*.so")

if len(soFileList) > 0:
    print("Find Plugin %s!"%soFileList)
else:
    print("No Plugin!")
for soFile in soFileList:
    ctypes.cdll.LoadLibrary(soFile)

logger = trt.Logger(trt.Logger.VERBOSE)
trt.init_libnvinfer_plugins(logger, '')
PLUGIN_CREATORS = trt.get_plugin_registry().plugin_creator_list

for plugin_creator in PLUGIN_CREATORS:
    print(plugin_creator.name, plugin_creator.plugin_namespace, plugin_creator.plugin_version)





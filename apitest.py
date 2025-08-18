from fastapi import FastAPI, Body
import uvicorn
from pydantic import BaseModel
from typing import Any, Dict, List
from ebeam import beam
from beamline import *
import beamline
from schematic import *
from fastapi.middleware.cors import CORSMiddleware
import inspect
import json
import importlib
import io
import base64
import pandas as pd
from excelElements import ExcelElements

ORIGINS = ["http://localhost:5173", "localhost:5173"]
moduleName = 'beamline'

class BeamlineInfo(BaseModel):
    #__root__: Dict[str, Dict[str, Any]]
    segmentName: str
    parameters: Dict[str, Any]

class LineAxObject(BaseModel):
    axis: str # temporary placeholder axes
    twiss: str
    x_axis: list[float]
    beamsegment: list

class AxesPNGData(BaseModel):
    images: Dict[float, Any]
    line_graph: LineAxObject

app = FastAPI()
ebeam = beam()
# Allow requests from your frontend (CORS!)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,  # In production, use your frontend's exact origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def getPngObjFromBeamList(beamlist):
    beam_dist = ebeam.gen_6d_gaussian(0,[1,1,1,1,0.1,100], 1000) #DONT HARDCORD NUM_PARTICLES
    schem = draw_beamline()
    axList, lineAxObj = schem.plotBeamPositionTransform(beam_dist, beamlist, plot=False, apiCall=True, scatter=True, interval=1)
    fig = lineAxObj['axis'].figure
    buf = io.BytesIO()
    fig.savefig(buf, format="png",bbox_inches="tight")
    buf.seek(0)
    lineAx_img = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()

    images = {}
    for index, axes in axList.items():

        fig = axes.figure
        buf = io.BytesIO()
        fig.savefig(buf, format="png",bbox_inches="tight")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()

        images.update({index: img_base64})
     
    lineAxObj['axis'] = lineAx_img
    lineAxObj['twiss'] = lineAxObj['twiss'].to_json()
    beamsegmentJson = []
    #for segment in lineAxObj['beamsegment']:
    #    beamsegmentJson.append(segment.__dict__)
    lineAxObj['beamsegment'] = beamsegmentJson
    #print(beamsegmentJson)

    lineAxObj = LineAxObject(**lineAxObj)
    pngObject = AxesPNGData(**{'images': images, 'line_graph': lineAxObj})
    return pngObject

def beamlineToJson():
    pass

@app.get("/")
def root():
    return {"Hello" : "World!"}

@app.post("/get-dist")
def gen_beam(particle_num : int): 
    beam_dist = ebeam.gen_6d_gaussian(0,[1,1,1,1,0.1,100], particle_num).tolist()
    return beam_dist

@app.post("/excel-to-beamline")
def excelToBeamline(excelJson: list[Dict[str, Any]]) -> list[dict[str, dict[str, Any]]]:
    excelHandler = ExcelElements(excelJson)
    beamlist = excelHandler.create_beamline()

    jsonBeamlist = []

    for segment in beamlist:
        clas = segment.__class__
        className = clas.__name__
        classSig= inspect.signature(clas.__init__)
        print(segment.__dict__)

        paramsDict = {}
        for name, param in classSig.parameters.items():
            if name == "self":
                continue
            paramVal = getattr(segment, name, None)
            paramsDict.update({name: paramVal})
                
        jsonBeamlist.append({className: paramsDict})

    return jsonBeamlist

@app.post("/axes")
def loadAxes(beamlineData: list[BeamlineInfo]) -> AxesPNGData:
    beamline = importlib.import_module("beamline")
    beamlist = []
    for segment in beamlineData:
        if hasattr(beamline, segment.segmentName):
            segmentClass = getattr(beamline, segment.segmentName)
            beamlist.append(segmentClass(**segment.parameters))

    pngObject = getPngObjFromBeamList(beamlist)
    return pngObject


@app.get("/beamsegmentinfo")
def getBeamSegmentInfo():
    module = importlib.import_module(moduleName)
    classes = inspect.getmembers(module, inspect.isclass)
    classes_in_module = [cls for name, cls in classes if cls.__module__ == moduleName and cls.__name__ not in ["beamline", "lattice"]]
    beamSegInfo = {}

    for cls in classes_in_module:
        sig = inspect.signature(cls.__init__)
        params_info = {}
    
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            default = (
                param.default
                if param.default is not inspect.Parameter.empty
                else 1  # or some other marker
            )
            params_info[name] = default

        params_info['color'] = cls.color  # Manually add class info about beam's color
    
        beamSegInfo[cls.__name__] = params_info

    return beamSegInfo

# Don't use, doesn't check for changes and server reloads
#if __name__ == "__main__":
    #uvicorn.run(app, host="127.0.0.1", port=8000)

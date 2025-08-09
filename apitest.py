from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict
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

ORIGINS = ["*", "http://localhost:5173", "localhost:5173"]
moduleName = 'beamline'

class BeamlineInfo(BaseModel):
    #__root__: Dict[str, Dict[str, Any]]
    segmentName: str
    parameters: Dict[str, Any]

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


@app.get("/")
def root():
    return {"Hello" : "World!"}

@app.post("/get-dist")
def gen_beam(particle_num : int): 
    beam_dist = ebeam.gen_6d_gaussian(0,[1,1,1,1,0.1,100], particle_num).tolist()
    return beam_dist

@app.post("/load-axes")
def loadAxes(beamlineData: list[BeamlineInfo]):
    beamline = importlib.import_module("beamline")
    beamlist = []
    for segment in beamlineData:
        print("|"+segment.segmentName+"|")
        if hasattr(beamline, segment.segmentName):
            print("hasattr check")
            segmentClass = getattr(beamline, segment.segmentName)
            beamlist.append(segmentClass(**segment.parameters))
    print(beamlist)
    beam_dist = ebeam.gen_6d_gaussian(0,[1,1,1,1,0.1,100], 1000)
    schem = draw_beamline()
    axList, lineAx = schem.plotBeamPositionTransform(beam_dist, beamlist, plot=False, apiCall=True, scatter=True)
    fig = lineAx.figure
    buf = io.BytesIO()
    fig.savefig(buf, format="png",bbox_inches="tight")
    buf.seek(0)
    lineAx_img = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()

    images = []
    for axes in axList:

        fig = axes.figure
        buf = io.BytesIO()
        fig.savefig(buf, format="png",bbox_inches="tight")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()

       # for axis in axes:
       #     fig = axis.figure
       #     buf = io.BytesIO()
       #     fig.savefig(buf, format="png",bbox_inches="tight")
       #     buf.seek(0)
       #     img_base64 = base64.b64encode(buf.read()).decode("utf-8")
       #     buf.close()
       #     newAxes.append(img_base64)
        images.append(img_base64)
    return {'images': images, 'line-graph': lineAx_img}


@app.get("/get-beamsegmentinfo")
def getBeamSegmentInfo():
    module = importlib.import_module(moduleName)
    classes = inspect.getmembers(module, inspect.isclass)
    classes_in_module = [cls for name, cls in classes if cls.__module__ == moduleName and cls.__name__ not in ["beamline", "lattice"]]
    beamSegInfo = {}
    #for cls in classes_in_module:
    #    sig = inspect.signature(cls.__init__)
    #    beamSegInfo.update({cls.__name__:
    #                        {p for p in sig.parameters if p != "self"}})
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
    
        beamSegInfo[cls.__name__] = params_info
    #json_string = json.dumps(beamSegInfo)
    #return json_string
    return beamSegInfo

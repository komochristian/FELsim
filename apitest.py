from fastapi import FastAPI
from ebeam import beam
from beamline import *
from fastapi.middleware.cors import CORSMiddleware
import inspect
import json
import importlib

ORIGINS = ["*", "http://localhost:5173"]
moduleName = 'beamline'

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

@app.post("/refresh-beamline")
def run_new_sim(beamline : list):
    pass

#@app.get("/twiss-data")
#def get_twiss(beamline: list):
#    pass

@app.get("/get-beamsegmentinfo")
def getBeamSegmentInfo():
    module = importlib.import_module(moduleName)
    classes = inspect.getmembers(module, inspect.isclass)
    classes_in_module = [cls for name, cls in classes if cls.__module__ == moduleName and cls.__name__ not in ["beamline", "lattice"]]
    beamSegInfo = []
    for cls in classes_in_module:
        sig = inspect.signature(cls.__init__)
        beamSegInfo.append({"name": cls.__name__,
                            "params": [p for p in sig.parameters if p != "self"]})
    #return [cls.__name__ for cls in classes_in_module if cls.__name__ not in ["lattice", "beamline"]]
    json_string = json.dumps(beamSegInfo)
    return json_string




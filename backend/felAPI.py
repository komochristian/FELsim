from fastapi import FastAPI, HTTPException
from pydantic import ValidationError
from typing import Any, List
from ebeam import beam
from beamline import *
from schematic import *
from fastapi.middleware.cors import CORSMiddleware
import inspect
import importlib
import io
import base64
from excelElements import ExcelElements
import uvicorn
import os
from dotenv import load_dotenv
import copy
import math
import numpy as np
from ApiSchemas import ExcelBeamlineElement, PlottingParameters, LineAxObject, AxesPNGData, GraphParameters, GraphPlotData, BeamSegmentsInfo

description = """
FEL beamline API used to interact with Python FEL and beamline simulation library.
"""

load_dotenv('../.env')  # Only during dev testing when not using Dockerfile...
FRONTEND_PORT = os.getenv('FRONTEND_PORT')

ORIGINS = [f'http://localhost:{FRONTEND_PORT}', f"localhost:{FRONTEND_PORT}"]
moduleName = 'beamline'

app = FastAPI(
    title="FEL Simulation API",
    description=description,
    version="1.0.0",
    contact={
        "name": "Christian Komo",
        "email": "komochristian@gmail.com",
    },
)
ebeam = beam()
# Allow requests from your frontend (CORS!)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,  # In production, use your frontend's exact origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def getPngObjFromBeamList(beamlist, plotParams: PlottingParameters):
    """
    Generates beamline simulation and returns base64 encoded images and twiss data.

    Parameters
    ----------
    - beamlist: List of beamline segments
    - plotParams: Object containing beamline and simulation parameters

    Returns
    -------
    - pngObject: Object containing base64 encoded particle plot images and twiss data
    """
    beam_dist = None
    # print(plotParams.beam_setup)
    if plotParams.beam_setup == 'twiss': beam_dist = ebeam.gen_6d_from_twiss(plotParams.twiss.model_dump(), plotParams.num_particles)
    else: beam_dist = ebeam.gen_6d_gaussian(0,[1,1,1,1,0.1,100], plotParams.num_particles)
    schem = draw_beamline()
    schem.DEFAULTINTERVALROUND = 10
    axList, lineAxObj = schem.plotBeamPositionTransform(beam_dist, beamlist, plot=False, apiCall=True, scatter=True, interval=plotParams.interval)

    images = {}
    for index, axes in axList.items():

        fig = axes.figure
        buf = io.BytesIO()
        fig.savefig(buf, format="png",bbox_inches="tight")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()

        images.update({index: img_base64})
     
    lineAxObj['twiss'] = lineAxObj['twiss'].to_json()

    lineAxObj = LineAxObject(**lineAxObj)
    pngObject = AxesPNGData(**{'images': images, 'line_graph': lineAxObj})
    return pngObject

@app.get("/")
def root():
    return {"FEL Beamline Simulation API"}

@app.post("/excel-to-beamline")
def excelToBeamline(excelJson: List[ExcelBeamlineElement]) -> List[BeamSegmentsInfo]:
    """
    Takes JSON formatted excel data and returns beamline object
    **Check Pydantic schema for data format

    Parameters
    ----------
    - excelJson: List of beamline elements from excel file

    Returns
    -------
    - beamline: List of beamline segments as dictionaries
    """
    try:
        excelJson_formatted = [item.model_dump(exclude_none=True, by_alias=True) for item in excelJson]
        excelHandler = ExcelElements(excelJson_formatted)
        beamlist = excelHandler.create_beamline()

        jsonBeamlist = []

        for segment in beamlist:
            clas = segment.__class__
            className = clas.__name__
            classSig= inspect.signature(clas.__init__)

            paramsDict = {}
            for name, param in classSig.parameters.items():
                if name == "self":
                    continue
                paramVal = getattr(segment, name, None)
                paramsDict.update({name: paramVal})
                    
            jsonBeamlist.append({className: paramsDict})
        beamlist_json_fixed = []
        for item in jsonBeamlist:
            for key, value in item.items():
                new_dict = {}
                new_dict.update(value)
                new_dict['name'] = key  # Temporary fix, will have to refactor to 'segment_type'
                beamlist_json_fixed.append(new_dict)
        # print(beamlist_json_fixed)
        return beamlist_json_fixed
    except ValidationError as e:
        print("Pydantic validation error:", e)
        return {"error": str(e)}
    except Exception as e:
        print("Error: ", e)
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/axes")
def loadAxes(plotParams: PlottingParameters) -> AxesPNGData:
    """
    Endpoint to return results of beamline simulation.
    Twiss data and particle plot images included.

    Parameters
    ----------
    -plotParams: Object containing beamline and simulation parameters

    Returns
    -------
    - pngObject: Object containing base64 encoded particle plot images and twiss data
    """
    try:
        beamline = importlib.import_module("beamline")
        beamlist = []
        beamlineData = plotParams.beamlineData
        for segment in beamlineData:
            if hasattr(beamline, segment.segmentName):
                segmentClass = getattr(beamline, segment.segmentName)
                beamlist.append(segmentClass(**segment.parameters))
    
        latObj = lattice(1)
        beamlist = latObj.changeBeamType(plotParams.beamType, plotParams.kineticE, beamlist)
        pngObject = getPngObjFromBeamList(beamlist, plotParams)        
        return pngObject
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")  # add this
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/beamsegmentinfo")
def getBeamSegmentInfo():
    """
    Returns most up to date beam segments available for beamline construction

    Returns
    -------
    beanSegInfo: Dictionary containing beam segment class names and their parameters with default values
    """
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
    for seg in beamSegInfo.values():
        seg.pop("name", None)
    return beamSegInfo

@app.post("/plot-parameters")
def plot_parameters(graphParams: GraphParameters) -> List[GraphPlotData]:
    """
    Returns twiss data as a function of different parameter values of a segment

    Parameters
    ----------
    graphParams: Object containing beamline and simulation parameters

    Returns
    -------
    plotInfo: List of objects containing twiss data plotted against parameter value
    """
    LABELMAPPING = {
        r'$\epsilon$ ($\pi$.mm.mrad)': 'emittance',
        r'$\alpha$': 'alpha',
        r'$\beta$ (m)': 'beta',
        r'$\gamma$ (rad/m)': 'gamma',
        r'$D$ (m)': 'dispersion',
        r'$D^{\prime}$': 'dispersion_prime',
        r'$\phi$ (deg)': 'angle',
        r'Envelope $E$ (mm)': 'envelope'
    }
    try:
        beamline_class = importlib.import_module(moduleName)
        beamlist = []
        beamlineData = graphParams.beamline_data
        for segment in beamlineData:
            if hasattr(beamline_class, segment.segmentName):
                segmentClass = getattr(beamline_class, segment.segmentName)
                beamlist.append(segmentClass(**segment.parameters))

        cleanedBeamlist = beamlist[:graphParams.beam_index]

        schem = draw_beamline()
        ebeam = beam()

        #  TODO: make the user able to configure the particle distribution
        beam_dist = ebeam.gen_6d_gaussian(0,[1,1,1,1,0.1,100], 1000)

        # print("Plotting initial beamline up to segment", cleanedBeamlist)

        #  100 chosen as a large number to speed up initial calculation
        schem.plotBeamPositionTransform(beam_dist, cleanedBeamlist, plot=False, interval=100, rendering=False)
        beam_dist = schem.matrixVariables

        beamObj = beamline(beamlist)
        indexOfSSegment = beamObj.findSegmentAtPos(graphParams.target_s_pos)

        newSegment = copy.deepcopy(beamObj.beamline[indexOfSSegment])
        newSegment.length = graphParams.target_s_pos - beamObj.beamline[indexOfSSegment - 1].endPos
        optimized_beamlist = beamObj.beamline[graphParams.beam_index:indexOfSSegment]
        optimized_beamlist.append(newSegment)

        # for i in optimized_beamlist:
        #     print("Printing segment:", i) 

        plotInfo = [] 
        domain_range = np.arange(graphParams.min, graphParams.max, graphParams.custom_step).tolist()
        if graphParams.max not in domain_range: domain_range.append(graphParams.max)

        for i in domain_range:
            setattr(optimized_beamlist[0], graphParams.target_parameter, i)
            twiss = schem.plotBeamPositionTransform(beam_dist, optimized_beamlist, plot=False, interval=100, rendering=False)
            # col = LABELMAPPING.get(graphParams.twiss_target, graphParams.twiss_target)
            plotDict = {f'parameter_value': i, 
                         'data': [
                                    {
                                        **{name: None if axis[-1] is None or math.isnan(axis[-1]) or math.isinf(axis[-1]) else axis[-1]
                                        for name, axis in twiss[col].items()},
                                        'twiss_parameter': LABELMAPPING.get(col, col)
                                    }
                                    for col in twiss.columns
                                 ]
                        }
            # print(plotDict)
            plotInfo.append(plotDict)

        # RETURN ALL TWISS, MAKE USER SELECT WHICH ONE
        return plotInfo
    
    except Exception as e:
        print(e)
        raise HTTPException(status_code=400, detail=str(e))

@app.post('/twiss-to-particles')
def getParticlesFromTwiss(twissParams):
    pass

# Don't use, doesn't check for changes and server reloads
#if __name__ == "__main__":
    #uvicorn.run(app, host="127.0.0.1", port=8000)

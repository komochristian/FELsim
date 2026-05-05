from fastapi.testclient import TestClient

from felAPI import app
import json

for route in app.routes:
    print(route.path, route.methods)

client = TestClient(app)

def assertDipole(response):
    dipole = response.json()["dipole"]

    # Assert keys exist
    assert "length" in dipole
    assert "angle" in dipole
    assert "fringeType" in dipole
    assert "color" in dipole

    # Assert types
    assert isinstance(dipole["length"], (int, float))
    assert isinstance(dipole["angle"], (int, float))
    assert isinstance(dipole["fringeType"], str)
    assert dipole["color"] == "forestgreen"

def assertDipoleWedge(response):
    dipole_wedge = response.json()["dipole_wedge"]
    # Assert keys exist
    expected_keys = {
        "length", "angle", "dipole_length", "dipole_angle",
        "pole_gap", "enge_fct", "fringeType", "color"
    }
    assert set(dipole_wedge.keys()) == expected_keys

    # Assert types
    assert isinstance(dipole_wedge["length"], (int, float))
    assert isinstance(dipole_wedge["angle"], (int, float))
    assert isinstance(dipole_wedge["dipole_length"], (int, float))
    assert isinstance(dipole_wedge["dipole_angle"], (int, float))
    assert isinstance(dipole_wedge["pole_gap"], (int, float))
    assert isinstance(dipole_wedge["enge_fct"], (int, float))
    assert isinstance(dipole_wedge["fringeType"], str)
    assert isinstance(dipole_wedge["color"], str)
    # Optionally, assert specific values
    assert dipole_wedge["color"] == "lightgreen"

def assertDriftLattice(response):
    drift = response.json()["driftLattice"]
    expected_keys = {"length", "color"}
    assert set(drift.keys()) == expected_keys
    assert isinstance(drift["length"], (int, float))
    assert isinstance(drift["color"], str)
    assert drift["color"] == "white"

def assertQpdLattice(response):
    qpd = response.json()["qpdLattice"]
    expected_keys = {"current", "length", "fringeType", "color"}
    assert set(qpd.keys()) == expected_keys
    assert isinstance(qpd["current"], (int, float))
    assert isinstance(qpd["length"], (int, float))
    assert isinstance(qpd["fringeType"], str)
    assert isinstance(qpd["color"], str)
    assert qpd["color"] == "lightcoral"

def assertQpfLattice(response):
    qpf = response.json()["qpfLattice"]
    expected_keys = {"current", "length", "fringeType", "color"}
    assert set(qpf.keys()) == expected_keys
    assert isinstance(qpf["current"], (int, float))
    assert isinstance(qpf["length"], (int, float))
    assert isinstance(qpf["fringeType"], str)
    assert isinstance(qpf["color"], str)
    assert qpf["color"] == "cornflowerblue"


def test_beamInfo():
    response = client.get("/beamsegmentinfo")

    assert response.status_code == 200
    
    assertDipole(response)
    assertDipoleWedge(response)
    assertDriftLattice(response)
    assertQpdLattice(response)
    assertQpfLattice(response)

def test_loadAxes():
    response = client.get("/axes")
    twiss_dict = json.loads(response.twiss)
    print(twiss_dict)

if __name__ == "__main__":
    test_beamInfo()
    test_loadAxes()
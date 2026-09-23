"""Regression tests for what PR #13 broke on main.

Each test drives the code the way its real caller does: the web API tests
send the payloads fel-app builds (App.jsx, ModalContent.jsx,
ExcelUploadButton.jsx), and the COSY tests construct the simulator the way
cosyAdapter does. The zero-momentum-spread Twiss test covers an older
division that the xsuite space-charge test from #14 runs into.

Author: Eremey Valetov
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import beamline
from beamline import alphaMagnetLattice, driftLattice, rfCavityLattice

# fel-app/src/constants.js
PRIVATEVARS = ['color', 'startPos', 'endPos', 'name', 'id', 'status']

# Colours before #13 (ce681ec); classes added by #13 keep theirs.
COLOURS = {
    'driftLattice': 'white',
    'qpfLattice': 'cornflowerblue',
    'qpdLattice': 'lightcoral',
    'dipole': 'forestgreen',
    'dipole_wedge': 'lightgreen',
    'alphaMagnetLattice': 'darkorange',
    'rfCavityLattice': 'gold',
}


@pytest.fixture(scope='module')
def api():
    pytest.importorskip('httpx')
    pytest.importorskip('fastapi')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import felAPI
    yield felAPI
    plt.close('all')


@pytest.fixture(scope='module')
def client(api):
    from fastapi.testclient import TestClient
    return TestClient(api.app, raise_server_exceptions=False)


@pytest.fixture(scope='module')
def segment_info(api):
    """The element defaults /beamsegmentinfo advertises, before serialisation."""
    return api.getBeamSegmentInfo()


def frontend_rows(info, names, lengths=None):
    """Table rows as App.jsx keeps them: class defaults plus s positions."""
    rows, s = [], 0.0
    for i, name in enumerate(names):
        row = {'name': name, **info[name]}
        if lengths and lengths[i] is not None:
            row['length'] = lengths[i]
        row['startPos'] = s
        s += row['length']
        row['endPos'] = s
        rows.append(row)
    return rows


def frontend_payload(rows):
    return [{'segmentName': r['name'],
             'parameters': {k: v for k, v in r.items() if k not in PRIVATEVARS}}
            for r in rows]


def test_beamsegmentinfo_colours(client):
    r = client.get('/beamsegmentinfo')
    assert r.status_code == 200, r.text
    info = r.json()
    assert set(COLOURS) <= set(info)
    for name, entry in info.items():
        assert isinstance(entry['color'], str), name
    for name, colour in COLOURS.items():
        assert info[name]['color'] == colour
        assert getattr(beamline, name).color == colour


def test_plot_parameters_frontend_payload(client, segment_info):
    names = ['driftLattice', 'qpfLattice', 'driftLattice', 'qpdLattice',
             'driftLattice', 'dipole', 'driftLattice']
    rows = frontend_rows(segment_info, names,
                         [0.5, None, 0.3, None, 0.3, None, 0.5])
    body = {
        'beam_index': 1,
        'target_parameter': 'current',
        'target_s_pos': rows[-1]['startPos'] + 0.2,
        'beamline_data': frontend_payload(rows),
        'min': 0,
        'max': 2,
        'custom_step': 1,
        'spread_data': {'beam_setup': 'import', 'data': None},
        'num_particles': 200,
    }
    r = client.post('/plot-parameters', json=body)
    assert r.status_code == 200, r.text
    scan = r.json()
    assert [p['parameter_value'] for p in scan] == [0, 1, 2]
    for point in scan:
        rows_by_param = {d['twiss_parameter']: d for d in point['data']}
        for plane in ('x', 'y'):
            assert math.isfinite(rows_by_param['beta'][plane])


def test_excel_to_beamline_frontend_json(client):
    # XLSX.utils.sheet_to_json output for three rows of
    # beam_excel/Beamline_elements.xlsx: header keys, empty cells omitted
    sheet = [
        {' Nomenclature': 'LIN.QPF.004', 'z start (m)': 0.358775,
         'z mid (m)': 0.403225, 'z end (m)': 0.447675,
         'Current A)': 0.885719309299156, 'Pole gap (m)': 0.027,
         'Element name': 'Quad', 'Channel #': 20, 'Sector': 'LIN',
         'Element': 'QPF'},
        {' Nomenclature': 'FC1.DPW.111', 'z start (m)': 11.099409,
         'z mid (m)': 11.104409, 'z end (m)': 11.109409,
         'Dipole Angle (deg)': 11.25, 'Dipole length (m)': 0.037389,
         'Dipole wedge (deg)': 0, 'Gap wedge (m)': 0.01,
         'Pole gap (m)': 0.0127,
         'Fringe Field Enge coefficients':
             ' 56.49, -50.79, 19.32, -3.621,  0.3315,  -0.01193',
         'Sector': 'FC1', 'Element': 'DPW'},
        {' Nomenclature': 'FC1.DPH.111', 'z start (m)': 11.109409,
         'z mid (m)': 11.1281035, 'z end (m)': 11.146798,
         'Dipole Angle (deg)': 11.25, 'Dipole length (m)': 0.037389,
         'Pole gap (m)': 0.0127, 'Element name': 'MkIII chicane dipole',
         'Sector': 'FC1', 'Element': 'DPH'},
    ]
    r = client.post('/excel-to-beamline', json=sheet)
    assert r.status_code == 200, r.text
    names = [seg['name'] for seg in r.json()]
    assert names == ['driftLattice', 'qpfLattice', 'driftLattice',
                     'dipole_wedge', 'dipole']


def test_rf_cavity_from_advertised_defaults(client, segment_info):
    defaults = {k: v for k, v in segment_info['rfCavityLattice'].items()
                if k not in PRIVATEVARS}
    cav = rfCavityLattice(**defaults)
    assert cav.voltage_mv is None and cav.gradient_mv_per_m is None
    assert 'no field' in str(cav)

    rows = frontend_rows(segment_info,
                         ['driftLattice', 'rfCavityLattice', 'driftLattice'],
                         [0.3, None, 0.3])
    r = client.post('/axes', json={
        'beamlineData': frontend_payload(rows), 'num_particles': 200,
        'beamType': 'electron', 'interval': 0.5, 'kineticE': 45,
        'spread_data': {'beam_setup': 'import', 'data': None}})
    assert r.status_code == 200, r.text

    cav = rfCavityLattice(3.048, 2856e6, gradient_mv_per_m=13.3)
    assert cav.voltage_mv == pytest.approx(13.3 * 3.048)


def test_lattice_file_rf_cavity_still_needs_a_field():
    import json
    from latticeLoaderBase import LatticeLoaderBase
    from tracked_dict import TrackedDict
    data = json.loads((BACKEND.parent / 'var' / 'slac_linac.json').read_text())
    line = LatticeLoaderBase(TrackedDict(data)).create_beamline()
    assert line[0].gradient_mv_per_m == pytest.approx(13.3)
    del data['beamline']['elements'][0]['parameters']['gradient_mv_per_m']
    with pytest.raises(ValueError, match='voltage_mv'):
        LatticeLoaderBase(TrackedDict(data)).create_beamline()


def test_alpha_magnet_round_trips_through_gui(client, segment_info):
    # App.jsx sums row lengths for s positions, so every element needs one
    advertised = segment_info['alphaMagnetLattice']
    assert advertised['length'] == pytest.approx(
        alphaMagnetLattice(advertised['current']).length)
    # length is keyword-only, so the label keeps its position from before
    assert alphaMagnetLattice(10.0, None, 'AM1').name == 'AM1'
    # The quad row fills every column, as some row of a real sheet does
    sheet = [
        {' Nomenclature': 'TST.AMG.001', 'z start (m)': 0.5, 'z mid (m)': 0.55,
         'z end (m)': 0.6, 'Current A)': 12.0, 'Element': 'AMG'},
        {' Nomenclature': 'TST.QPF.002', 'z start (m)': 0.8, 'z mid (m)': 0.84,
         'z end (m)': 0.8889, 'Current A)': 1.0, 'Dipole Angle (deg)': 0,
         'Dipole length (m)': 0, 'Dipole wedge (deg)': 0, 'Gap wedge (m)': 0,
         'Pole gap (m)': 0.027, 'Fringe Field Enge coefficients': '0',
         'Element name': 'Quad', 'Channel #': 20, 'Label': 'Q', 'Sector': 'TST',
         'Element': 'QPF'},
    ]
    r = client.post('/excel-to-beamline', json=sheet)
    assert r.status_code == 200, r.text
    rows = r.json()
    alpha = next(row for row in rows if row['name'] == 'alphaMagnetLattice')
    assert alpha['length'] == pytest.approx(alphaMagnetLattice(12.0).length)

    r = client.post('/axes', json={
        'beamlineData': frontend_payload(rows), 'num_particles': 200,
        'beamType': 'electron', 'interval': 0.5, 'kineticE': 45,
        'spread_data': {'beam_setup': 'import', 'data': None}})
    assert r.status_code == 200, r.text


def test_plot_parameters_refuses_stale_alpha_length(client, segment_info):
    # The table keeps an alpha magnet's length when its current is edited, so
    # positions after it no longer match the line the backend simulates.
    names = ['driftLattice', 'alphaMagnetLattice', 'driftLattice', 'qpfLattice',
             'driftLattice']
    rows = frontend_rows(segment_info, names, [0.3, None, 0.3, None, 0.5])

    def scan(rows):
        return client.post('/plot-parameters', json={
            'beam_index': 3, 'target_parameter': 'current',
            'target_s_pos': rows[-1]['startPos'] + 0.1,
            'beamline_data': frontend_payload(rows),
            'min': 0.5, 'max': 1.5, 'custom_step': 0.5,
            'spread_data': {'beam_setup': 'import', 'data': None},
            'num_particles': 200})

    r = scan(rows)
    assert r.status_code == 200, r.text

    rows[1]['current'] = 10.0
    r = scan(rows)
    assert r.status_code == 400
    real = alphaMagnetLattice(10.0).length
    assert 'alphaMagnetLattice' in r.json()['detail']
    assert f'{real:.9g}' in r.json()['detail']

    rows = frontend_rows(segment_info, names, [0.3, real, 0.3, None, 0.5])
    rows[1]['current'] = 10.0
    r = scan(rows)
    assert r.status_code == 200, r.text


def test_sliced_tracking_keeps_alpha_magnet_whole():
    # The GUI slices every element into steps of the plot interval; the alpha
    # map has no slices, so any step shorter than the orbit applied it again.
    from beamPropagator import propagate
    from schematic import draw_beamline
    line = [driftLattice(0.2), alphaMagnetLattice(10.0), driftLattice(0.2)]
    assert line[1].length > 0.5
    particles = np.random.default_rng(2).normal(size=(200, 6))
    whole = particles
    for seg in line:
        whole = seg.useMatrice(whole)

    sliced = list(propagate(line, particles, 0.1))[-1].particles
    np.testing.assert_allclose(sliced, whole, rtol=0, atol=1e-12)

    schem = draw_beamline()
    schem.plotBeamPositionTransform(particles, line, plot=False, interval=0.1,
                                    rendering=False)
    np.testing.assert_allclose(schem.matrixVariables, whole, rtol=0, atol=1e-12)


def test_cosy_simulator_constructs():
    # Construction only; no COSY INFINITY binary is needed.
    cosySimulator = pytest.importorskip('cosySimulator')
    sim = cosySimulator.COSYSimulator(excel_path=None, config_dict={})
    assert sim.cosy_dist_dir in sim.search_dirs
    sim = cosySimulator.COSYSimulator(excel_path=None, config_dict={},
                                      cosy_dist_dir='/opt/cosy')
    assert sim.cosy_dist_dir == '/opt/cosy'

    cosyAdapter = pytest.importorskip('cosyAdapter')
    adapter = cosyAdapter.COSYAdapter(mode='transfer_matrix')
    assert adapter.get_native_simulator().excel_path is None


def test_beamline_builder_without_file():
    from beamlineBuilder import BeamlineBuilder
    assert BeamlineBuilder(None).excel_path is None
    assert BeamlineBuilder().excel_path is None
    with pytest.raises(FileNotFoundError):
        BeamlineBuilder('no_such_beamline.xlsx')


@pytest.mark.parametrize('offset', [0.0, 0.3])
def test_twiss_finite_with_zero_momentum_spread(offset):
    from ebeam import beam
    rng = np.random.default_rng(1)
    dist = rng.normal(size=(500, 6))
    dist[:, 5] = offset
    _, _, twiss = beam().cal_twiss(dist)
    transverse = twiss.loc[['x', 'y']]
    assert np.all(np.isfinite(transverse.to_numpy()))
    # a constant offset leaves only roundoff in the variance
    assert np.all(transverse.iloc[:, 4:6].to_numpy() == 0.0)


def test_rftrack_raises_on_alpha_magnet():
    pytest.importorskip('RF_Track')
    from rftrackAdapter import RFTrackAdapter
    from simulatorBase import BeamlineElement
    sim = RFTrackAdapter(beam_energy=45.0)
    with pytest.raises(NotImplementedError):
        sim.set_beamline([driftLattice(0.1), alphaMagnetLattice(current=5.0),
                          driftLattice(0.1)])
    with pytest.raises(NotImplementedError):
        sim.set_beamline([BeamlineElement('ALPHA_MAGNET', 0.3, current=5.0)])


def test_rftrack_refuses_rf_cavity_without_field():
    pytest.importorskip('RF_Track')
    from rftrackAdapter import RFTrackAdapter
    sim = RFTrackAdapter(beam_energy=45.0)
    with pytest.raises(ValueError, match='gradient_mv_per_m'):
        sim.set_beamline([driftLattice(0.1), rfCavityLattice(3.0, 2856e6),
                          driftLattice(0.1)])


def test_xsuite_refuses_rf_cavity_without_field():
    pytest.importorskip('xtrack')
    pytest.importorskip('xpart')
    from xsuiteAdapter import XsuiteAdapter
    sim = XsuiteAdapter(beam_energy=45.0)
    sim.set_beamline([driftLattice(0.1), rfCavityLattice(3.0, 2856e6),
                      driftLattice(0.1)])
    particles = np.random.default_rng(3).normal(scale=1e-3, size=(20, 6))
    with pytest.raises(ValueError, match='gradient_mv_per_m'):
        sim.simulate(particles)


def test_xsuite_raises_on_alpha_magnet():
    pytest.importorskip('xtrack')
    pytest.importorskip('xpart')
    from xsuiteAdapter import XsuiteAdapter
    sim = XsuiteAdapter(beam_energy=45.0)
    sim.set_beamline([driftLattice(0.1), alphaMagnetLattice(current=5.0),
                      driftLattice(0.1)])
    particles = np.random.default_rng(3).normal(scale=1e-3, size=(20, 6))
    with pytest.raises(NotImplementedError):
        sim.simulate(particles)

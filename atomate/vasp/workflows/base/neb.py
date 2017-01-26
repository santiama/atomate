#!/usr/bin/env python
# coding: utf-8

"""
This module defines the Climbing Image Nudged Elastic Band (CI-NEB) workflow.
1) initial relaxation fireworks (pre-defined)
2) Generate endpoints --> Two endpoints relaxation
3) Use two endpoints --> Generate images --> CI-NEB
4)                       Images --> CI-NEB
"""

import yaml
import os
from datetime import datetime

from monty.serialization import loadfn
from pymatgen.core import Structure
from pymatgen.io.vasp.sets import MPRelaxSet, MPStaticSet
from pymatgen_diffusion.neb.io import MVLCINEBEndPointSet, MVLCINEBSet, get_endpoints_from_index
from fireworks.core.firework import Firework, Workflow
from fireworks.core.launchpad import LaunchPad
from atomate.utils.utils import get_logger
from atomate.utils.utils import get_wf_from_spec_dict

from atomate.vasp.fireworks.core import OptimizeFW, NEBFW, NEBRelaxationFW

__author__ = "Hanmei Tang, Iek-Heng Chu"
__email__ = 'hat003@eng.ucsd.edu, ihchu@eng.ucsd.edu'

logger = get_logger(__name__)

# module_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))

# the template of fw_spec, 'mandatory' means the value must be validated with carefulness.
spec_orig = {"neb_id": 0,  # considering....
             "path_id": 0,  # considering....
             "wfname": "CINEB",  # optional
             "vasp_cmd": ">>vasp_cmd<<",  # optional, 'default vasp'
             "gamma_vasp_cmd": ">>gamma_vasp_cmd<<",  # optional, default 'vasp_gam'
             "mpi_command": {"command": "mpirun", "np_tag": "-np"},  # mandatory, option limited to -np tag only
             "ppn": "24",  # Processors per node  # mandatory
             "db_file": ">>db_file<<",  # TODO: May remove this  # optional
             "_category": "tscc-atomate",  # mandatory setting from fireworks
             "_queueadapter": {"nnodes": 1},  # mandatory, default
             "calc_locs": "",  # unnecessary, update during runtime
             "rlx_dir": "",  # unnecessary, update during runtime
             "ep0_dir": "",  # TODO: This is enveloped in CalLocs
             "ep1_dir": "",  # unnecessary, update during runtime
             "neb_dir": {},  # key = "1", "2" ... are path_id
             "rlx_st": {},  # unnecessary, update during runtime
             "ep0_st": {},  # mandatory if started using endpoints
             "ep1_st": {},  # unnecessary, update during runtime
             "path_sites": [],
             # mandatory if started using images
             "images": []}  # otherwise unnecessary, update during runtime


def _update_spec_from_inputs(spec,
                             structure=None, path_sites=None,
                             endpoints=None, images=None):
    """
    Update spec according to inputs.

    Args:
        spec(dict): original spec
        structure (Structure/dict): perfect cell structure.
        path_sites ([int, int]): Indicating pathway site indexes.
        endpoints ([Structure/dict]): The two endpoints [ep0, ep1].
        images ([s0_dict, s1_dict, ...]): The image structures,
            including the two endpoints.
    """
    s = spec_orig.copy()
    s.update(spec)

    if structure is not None:
        if isinstance(structure, Structure):
            s["rlx_st"] = structure.as_dict()
        elif isinstance(structure, dict):
            s["rlx_st"] = structure
        else:
            raise TypeError("Unable to parse the given structure!")

    if path_sites is not None:
        if isinstance(path_sites, list) and len(path_sites) == 2:
            s["path_sites"] = path_sites
        else:
            raise TypeError("path_sites should be a list!")

    if endpoints is not None:
        try:
            s["ep0_st"] = endpoints[0].as_dict()
            s["ep1_st"] = endpoints[1].as_dict()
        except:
            s["ep0_st"] = endpoints[0]
            s["ep1_st"] = endpoints[1]

    if images is not None:
        if len(images) <= 2:
            raise ValueError("Too few images!")
        s["images"] = images
        n_images = len(images) - 2
        s["_queueadapter"].update({"nnodes": str(n_images)})

    return s


def _get_mpi_command(spec, vasp):  # TODO: Enable setting using >>my_vasp_cmd<<
    """
    A convenience method to get neb command using mpi program:
    E.g.: 'mpirun -np 48 vasp'

    Args:
        spec (dict): fw_spec
        vasp (str): choose from ["std", "gam"].

    Returns:
        mpi command (str).
    """
    if vasp == "std":
        exe = spec["vasp_cmd"]
    elif vasp == "gam":
        exe = spec["gamma_vasp_cmd"]
    else:
        raise ValueError("Choose mode from \'std\' and \'gam\'!")

    nnodes = spec["_queueadapter"]["nnodes"]
    ppn = spec["ppn"]
    ncpu = int(nnodes) * int(ppn)

    mpi_cmd = spec["mpi_command"]["command"]
    np_tag = spec["mpi_command"]["np_tag"]
    full_mpi_command = "{} {} {} {}".format(mpi_cmd, np_tag, ncpu, exe)

    return full_mpi_command


def get_wf_neb_from_structure(structure, path_sites,
                              is_optimized=True,
                              wfname=None, neb_round=1,
                              spec=None,
                              uis_ini=None, uis_ep=None,
                              uis_neb=None):
    """
    Get a CI-NEB workflow from given perfect structures.
    Workflow: (Init relax) -- Endpoints relax -- NEB_1 -- NEB_2 - ... - NEB_r
              mode 1: rlx--ep--neb(r)
              mode 2: ep--neb(r)
    Args:
        structure (Structure): The perfect structure.
        path_sites (list[int]): The two vacancy site indices.
        is_optimized (bool): True implies the provided structure is optimized,
            otherwise run initial relaxation before generating endpoints.
        wfname (str): some appropriate name for the workflow.
        neb_round (int): Times of NEB calculations.
        spec (dict): user setting spec settings to overwrite spec_orig.
        uis_ini (dict): Additional user_incar_settings for initial relaxations.
        uis_ep (dict): Additional user_incar_settings for endpoint relaxations.
        uis_neb (list of dict): Additional user_incar_settings for NEB.
                        Eg.: [{"IOPT": 7}, {"IOPT": 1}, {"EDIFFG": -0.02}]

    Returns:
        Workflow
    """
    logger.info("Getting workflow from perfect structure.")

    # Read in params.
    formula = structure.composition.reduced_formula
    wfname = wfname or datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S-%f')
    spec = spec or {}

    uis_ini = uis_ini or {}
    uis_ep = uis_ep or {}
    uis_neb = uis_neb or []
    vasp_cmd = _get_mpi_command(spec, "std")
    gamma_vasp_cmd = _get_mpi_command(spec, "gam")
    endpoints = get_endpoints_from_index(structure, path_sites)
    eps_dict = [e.as_dict() for e in endpoints]  # pseudo endpoints

    # Get neb fireworks.
    neb_fws = []
    for n in range(neb_round):
        fw = NEBFW(spec=spec,
                   name=formula,
                   neb_label=str(n+1),
                   from_images=False,
                   vasp_input_set=None,
                   user_incar_settings=uis_neb[n],
                   vasp_cmd=vasp_cmd,
                   gamma_vasp_cmd=gamma_vasp_cmd,
                   cust_args={})
        neb_fws.append(fw)

    # Get relaxation fireworks.
    if is_optimized:
        spec = _update_spec_from_inputs(spec, endpoints=eps_dict)
        rlx_fws = [NEBRelaxationFW(spec=spec,
                                   st_label="ep{}".format(i),
                                   name=formula,
                                   vasp_input_set=None,
                                   user_incar_settings=uis_ep,
                                   vasp_cmd=vasp_cmd,
                                   gamma_vasp_cmd=gamma_vasp_cmd,
                                   cust_args={}) for i in [0, 1]]

        links = {rlx_fws[0]: [neb_fws[0]],
                 rlx_fws[1]: [neb_fws[0]]}
    else:
        spec = _update_spec_from_inputs(spec,
                                        structure=structure,
                                        path_sites=path_sites,
                                        endpoints=eps_dict)

        rlx_fws = [NEBRelaxationFW(spec=spec,
                                   st_label="{}".format(label),
                                   name=formula,
                                   vasp_input_set=None,
                                   user_incar_settings=incar)
                   for label, incar in zip(["rlx", "ep0", "ep1"],
                                           [uis_ini, uis_ep, uis_ep])]

        links = {rlx_fws[0]: [rlx_fws[1], rlx_fws[2]],
                 rlx_fws[1]: [neb_fws[0]],
                 rlx_fws[2]: [neb_fws[0]]}

    # Append Firework links.
    fws = rlx_fws + neb_fws
    if neb_round > 1:
        for r in range(1, neb_round):
            links[neb_fws[r-1]] = [neb_fws[r]]

    workflow = Workflow(fws, links_dict=links,
                        name="neb_{}".format(wfname))
    return workflow


def get_wf_neb_from_endpoints(endpoints=None,
                              is_optimized=True,
                              wfname=None,
                              spec=None, neb_round=1,
                              uis_ep=None, uis_neb=None):
    """
    Get a CI-NEB workflow from given endpoints.
    Workflow: (Endpoints relax -- ) NEB_1 -- NEB_2 - ... - NEB_r
              mode 3: ep--neb(r)
              mode 4: neb(r)

    Args:
        endpoints (list[Structure]): The image structures,
            if None then read from spec.
        is_optimized (bool): True implies the provided endpoint structures
            are optimized, otherwise run endpoints relaxation before NEB.
        wfname (str): some appropriate name for the workflow.
        spec (dict): user setting spec settings to overwrite spec_orig.
        neb_round (int): Times of NEB calculations.
        uis_ep (dict): Additional user_incar_settings for endpoint relaxations.
        uis_neb (list of dict): Additional user_incar_settings for NEB.
                        Eg.: [{"IOPT": 7}, {"IOPT": 1}, {"EDIFFG": -0.02}]

    Returns:
        Workflow
    """
    logger.info("Getting workflow from endpoints.")

    # Read in params.
    if endpoints is not None:
        endpoints_dict = [s.as_dict() for s in endpoints]
    else:
        endpoints_dict = [spec["ep0_st"], spec["ep1_st"]]
    formula = endpoints[0].composition.reduced_formula
    wfname = wfname or datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S-%f')
    spec = spec or {}
    spec = _update_spec_from_inputs(spec, endpoints=endpoints_dict)
    uis_ep = uis_ep or {}
    uis_neb = uis_neb or []
    vasp_cmd = _get_mpi_command(spec, "std")
    gamma_vasp_cmd = _get_mpi_command(spec, "gam")

    neb_fws = []
    for n in range(neb_round):
        fw = NEBFW(spec=spec,
                   name=formula,
                   neb_label=str(n+1),
                   from_images=False,
                   vasp_input_set=None,
                   user_incar_settings=uis_neb[n],
                   vasp_cmd=vasp_cmd,
                   gamma_vasp_cmd=gamma_vasp_cmd,
                   cust_args={})
        neb_fws.append(fw)

    workflow = Workflow(neb_fws, name="neb_workflow_{}".format(wfname))

    # Add endpoints relaxation if structures not optimized.
    if not is_optimized:
        ep_fws = [NEBRelaxationFW(spec=spec,
                                  st_label="ep{}".format(i),
                                  name=formula,
                                  vasp_input_set=None,
                                  user_incar_settings=uis_ep,
                                  vasp_cmd=vasp_cmd,
                                  gamma_vasp_cmd=gamma_vasp_cmd,
                                  cust_args={}) for i in [0, 1]]
        # Create Firework links.
        fws = ep_fws + neb_fws
        links = {ep_fws[0]: [neb_fws[0]],
                 ep_fws[1]: [neb_fws[0]]}
        if neb_round > 1:
            for r in range(1, neb_round):
                links[neb_fws[r-1]] = [neb_fws[r]]

        workflow = Workflow(fws, links_dict=links,
                            name="neb_{}".format(wfname))
    return workflow


def get_wf_neb_from_images(images=None,
                           wfname=None,
                           spec=None,
                           neb_round=1,
                           uis_neb=None):
    """
    Get a CI-NEB workflow from given images.
    Workflow: NEB_1 -- NEB_2 - ... - NEB_n

    Args:
        images (list[Structure]): The image structures,
            if None then read from spec["images"]
        wfname (str): some appropriate name for the workflow.
        spec (dict): user setting spec settings to overwrite spec_orig.
        neb_round (int): Times of NEB calculations.
        uis_neb (list of dict): Additional user_incar_settings for NEB.
                        Eg.: [{"IOPT": 7}, {"IOPT": 1}, {"EDIFFG": -0.02}]

    Returns:
        Workflow
    """
    logger.info("Getting workflow from images.")

    # Read in params.
    spec = spec or {}
    formula = images[0].composition.reduced_formula
    if images is not None:
        images_dict = [s.as_dict() for s in images]
    else:
        images_dict = spec["images"]  # TODO: what if spec is {}
    wfname = wfname or datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S-%f')
    spec = _update_spec_from_inputs(spec, images=images_dict)
    uis_neb = uis_neb or []
    vasp_cmd = _get_mpi_command(spec, "std")
    gamma_vasp_cmd = _get_mpi_command(spec, "gam")

    fws = []
    for n in range(neb_round):
        fw = NEBFW(spec=spec,
                   name=formula,
                   neb_label=str(n + 1),
                   from_images=True,
                   vasp_input_set=None,
                   user_incar_settings=uis_neb[n],
                   vasp_cmd=vasp_cmd,
                   gamma_vasp_cmd=gamma_vasp_cmd,
                   cust_args={})
        fws.append(fw)

    workflow = Workflow(fws, name="neb_{}".format(wfname))

    return workflow


# def test_get_wf_neb_from_images():
#     # test_dir = "/home/hat003/repos/atomate/atomate/vasp/" \
#     #            "workflows/tests/test_files/neb_wf/1/inputs"
#     test_dir = "/Users/hanmeiTang/repos/atomate/atomate/vasp" \
#                "/workflows/tests/test_files/neb_wf/1/inputs"
#     images = [Structure.from_file(os.path.join(test_dir, "{:02d}/POSCAR".format(i)))
#               for i in range(5)]
#     wfname = "images_wf"
#
#     wf = get_wf_neb_from_images(images, wfname, neb_round=1)
#
#     launchpad = LaunchPad.from_file(os.path.join(os.environ['HOME'],
#                                                  '.fireworks',
#                                                  'my_launchpad.yaml'))
#     launchpad.add_wf(wf)


if __name__ == "__main__":
    from pymatgen.util.testing import PymatgenTest

    # test_get_wf_neb_from_images()
    # structures = [PymatgenTest.get_structure("Si")]
    # wf = get_wf_neb(structures)
    pass

#!/usr/bin/env python
# coding: utf-8

"""
This module defines the Climbing Image Nudged Elastic Band (CI-NEB) workflow.
"""

import os
from datetime import datetime

from pymatgen.core import Structure
from pymatgen_diffusion.neb.io import get_endpoints_from_index
from fireworks.core.firework import Firework, Workflow

from atomate.vasp.fireworks.core import NEBFW, NEBRelaxationFW

__author__ = "Hanmei Tang, Iek-Heng Chu"
__email__ = 'hat003@eng.ucsd.edu, ihchu@eng.ucsd.edu'

# the template of fw_spec
# 'mandatory' means the value must be validated with carefulness.
spec_orig = {"neb_id": 0,  # considering....
             "path_id": 0,  # considering....
             "wf_name": "CINEB",  # optional
             "vasp_cmd": ">>vasp_cmd<<",  # optional, 'default vasp'
             "gamma_vasp_cmd": ">>gamma_vasp_cmd<<",  # optional, default 'vasp_gam'
             "mpi_command": {"command": "mpirun", "np_tag": "-np"},  # mandatory, option limited to -np tag only
             "ppn": "24",  # Processors per node  # mandatory
             "db_file": ">>db_file<<",  # TODO: May remove this  # optional
             "_category": "tscc-atomate",  # mandatory setting from fireworks
             "_queueadapter": {"nnodes": 1},  # mandatory, default
             "calc_locs": [],  # unnecessary, update during runtime
             "source_dir": os.environ["PWD"],
             "ini_st": {},  # unnecessary, update during runtime
             "ep0_st": {},  # mandatory if started using endpoints
             "ep1_st": {},  # unnecessary, update during runtime
             "path_sites": [],
             # mandatory if started using images
             "neb": [[{}]]
             }  # otherwise unnecessary, update during runtime


def _update_spec_from_inputs(spec=None, wf_name=None,
                             structure=None, path_sites=None,
                             endpoints=None, images=None):
    """
    Update spec according to inputs.

    Args:
        spec(dict): original spec
        structure (Structure/dict): perfect cell structure.
        path_sites ([int, int]): Indicating pathway site indexes.
        endpoints ([Structure/dict]): The two endpoints [ep0, ep1].
        images ([s0_dict, s1_dict, ...]): list of image_dict,
            including the two endpoints.
    """
    spec = spec or {}
    s = spec_orig.copy()
    s.update(spec)

    if structure is not None:
        if isinstance(structure, Structure):
            s["ini_st"] = structure.as_dict()
        elif isinstance(structure, dict):
            s["ini_st"] = structure
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
        assert isinstance(images, list)
        neb = s.get("neb")
        if len(neb) >= 1:  # at least one round of neb
            assert len(neb[0]) == len(images)
        else:  # no neb record
            assert len(images) >= 3
        neb.append(images)
        s["neb"] = neb
        n_images = len(images) - 2
        s["_queueadapter"].update({"nnodes": str(n_images)})

    if wf_name is not None:
        s["wf_name"] = wf_name

    return s


# TODO: Enable setting using >>my_vasp_cmd<<
def _get_mpi_command(spec, vasp):
    """
    A convenience method to get neb command using mpi program:
    E.g.: 'mpirun -np 48 vasp'

    Args:
        spec (dict): fw_spec
        vasp (str): choose from ["std", "gam"].

    Returns:
        mpi command (str).
    """
    assert vasp in ["std", "gam"]
    exe = spec["vasp_cmd"] if vasp == "std" else spec["gamma_vasp_cmd"]

    nnodes = spec["_queueadapter"]["nnodes"]
    ppn = spec["ppn"]
    ncpu = int(nnodes) * int(ppn)

    mpi_cmd = spec["mpi_command"]["command"]
    np_tag = spec["mpi_command"]["np_tag"]
    full_mpi_command = "{} {} {} {}".format(mpi_cmd, np_tag, ncpu, exe)

    return full_mpi_command


def _get_incar(mode, user_incar_settings=None):
    """
    Get user_incar_settings for every step.
    Args:
        mode (str): choose from "parent", "endpoints", "NEB"
            "parent": [parent_dict, endpoints_dict, neb_dict1, neb_dict2, ...]
            "endpoints": [endpoints_dict, neb_dict1, neb_dict2, ...]
            "NEB": [neb_dict1, neb_dict2, ...]
        user_incar_settings ([dict]): list of dict to be parsed.
    Returns:
        user_incar_settings (dict):
            (uis_ini, uis_ep, uis_neb), in which uis_ini and uis_ep are dict
            and uis_neb is a list of dict.
    """
    # Validate input type
    assert mode in ["parent", "endpoints", "NEB"]
    assert isinstance(user_incar_settings, list)
    for incar in user_incar_settings:
        assert isinstance(incar, dict)

    if mode == "parent":
        assert len(user_incar_settings) >= 3
        uis_ini = user_incar_settings[0]
        uis_ep = user_incar_settings[1]
        uis_neb = user_incar_settings[2:]
    elif mode == "endpoints":
        assert len(user_incar_settings) >= 2
        uis_ini = {}
        uis_ep = user_incar_settings[0]
        uis_neb = user_incar_settings[1:]
    else:
        assert len(user_incar_settings) >= 1
        uis_ini, uis_ep = {}, {}
        uis_neb = user_incar_settings

    return uis_ini, uis_ep, uis_neb


def get_wf_neb_from_structure(structure, path_sites, user_incar_settings,
                              is_optimized=True, wf_name=None, spec=None):
    """
    Get a CI-NEB workflow from given perfect structures.
    Workflow: (Init relax) -- Endpoints relax -- NEB_1 -- NEB_2 - ... - NEB_r
              mode 1: rlx--ep--neb(r)
              mode 2: ep--neb(r)
    Args:
        structure (Structure): The perfect structure.
        path_sites (list[int]): The two vacancy site indices.
        user_incar_settings([dict]): Additional user_incar_settings
            corresponded with fw
        is_optimized (bool): True implies the provided structure is optimized,
            otherwise run initial relaxation before generating endpoints.
        wf_name (str): some appropriate name for the workflow.
        spec (dict): user setting spec settings to overwrite spec_orig.

    Returns:
        Workflow
    """
    # Get names
    formula = structure.composition.reduced_formula
    wf_name = wf_name or datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S-%f')

    # Get INCARs
    mode = "endpoints" if is_optimized else "parent"
    uis_ini = _get_incar(mode, user_incar_settings)[0]
    uis_ep = _get_incar(mode, user_incar_settings)[1]
    uis_neb = _get_incar(mode, user_incar_settings)[2]
    neb_round = len(uis_neb)

    # Get relaxation fireworks.
    if is_optimized:  # Start from endpoints
        endpoints = get_endpoints_from_index(structure, path_sites)
        endpoints_dict = [e.as_dict() for e in endpoints]
        spec = _update_spec_from_inputs(spec, structure=structure,
                                        path_sites=path_sites,
                                        wf_name=wf_name,
                                        endpoints=endpoints_dict)

        # Get mpi vasp command
        vasp_cmd = _get_mpi_command(spec, "std")
        gamma_vasp_cmd = _get_mpi_command(spec, "gam")

        # Get relax fireworks
        rlx_fws = [NEBRelaxationFW(spec=spec,
                                   st_label=i,
                                   name=formula,
                                   vasp_input_set=None,
                                   user_incar_settings=uis_ep,
                                   vasp_cmd=vasp_cmd,
                                   gamma_vasp_cmd=gamma_vasp_cmd,
                                   cust_args={}) for i in ["ep0", "ep1"]]

        links = {rlx_fws[0]: [neb_fws[0]],
                 rlx_fws[1]: [neb_fws[0]]}
    else:
        spec = _update_spec_from_inputs(spec,
                                        wf_name=wf_name,
                                        path_sites=path_sites)
        vasp_cmd = _get_mpi_command(spec, "std")
        gamma_vasp_cmd = _get_mpi_command(spec, "gam")

        ini_fw = NEBRelaxationFW(spec=spec,
                                 st_label="ini", name=formula,
                                 vasp_input_set=None,
                                 user_incar_settings=uis_ini)





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

    # Get neb fireworks.
    neb_fws = []
    for n in range(neb_round):
        fw = NEBFW(spec=spec,
                   name=formula,
                   neb_label=str(n + 1),
                   from_images=False,
                   user_incar_settings=uis_neb[n],
                   vasp_cmd=vasp_cmd,
                   gamma_vasp_cmd=gamma_vasp_cmd,
                   cust_args={})
        neb_fws.append(fw)

    # Append Firework links.
    fws = rlx_fws + neb_fws
    if neb_round > 1:
        for r in range(1, neb_round):
            links[neb_fws[r - 1]] = [neb_fws[r]]

    workflow = Workflow(fws, links_dict=links,
                        name="neb_{}".format(wf_name))
    return workflow


def get_wf_neb_from_endpoints(user_incar_settings, endpoints=None,
                              is_optimized=True, wf_name=None, spec=None):
    """
    Get a CI-NEB workflow from given endpoints.
    Workflow: (Endpoints relax -- ) NEB_1 -- NEB_2 - ... - NEB_r
              endpoints not optimized: ep--neb(r)
              endpoints are optimized: neb(r)

    Args:
        user_incar_settings([dict]): Additional user_incar_settings
            corresponded with fw
        endpoints (list[Structure]): The image structures,
            if None then read from spec.
        is_optimized (bool): True implies the provided endpoint structures
            are optimized, otherwise run endpoints relaxation before NEB.
        wf_name (str): some appropriate name for the workflow.
        spec (dict): user setting spec settings to overwrite spec_orig.

    Returns:
        Workflow
    """
    if endpoints is not None:
        for e in endpoints:
            assert isinstance(e, Structure)
        endpoints_dict = [s.as_dict() for s in endpoints]
    else:
        endpoints_dict = [spec["ep0_st"], spec["ep1_st"]]

    formula = endpoints[0].composition.reduced_formula
    wf_name = wf_name or datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S-%f')

    mode = "NEB" if is_optimized else "endpoints"
    uis_ep = _get_incar(mode, user_incar_settings)[1]
    uis_neb = _get_incar(mode, user_incar_settings)[2]
    neb_round = len(uis_neb)

    spec = _update_spec_from_inputs(spec, endpoints=endpoints_dict)
    vasp_cmd = _get_mpi_command(spec, "std")
    gamma_vasp_cmd = _get_mpi_command(spec, "gam")

    neb_fws = []
    for n in range(neb_round):
        fw = NEBFW(spec=spec,
                   name=formula,
                   neb_label=str(n + 1),
                   from_images=False,
                   user_incar_settings=uis_neb[n],
                   vasp_cmd=vasp_cmd,
                   gamma_vasp_cmd=gamma_vasp_cmd,
                   cust_args={})
        neb_fws.append(fw)

    workflow = Workflow(neb_fws, name=wf_name)

    # Add endpoints relaxation if structures not optimized.
    if not is_optimized:
        ep_fws = [NEBRelaxationFW(spec=spec,
                                  st_label=i,
                                  name=formula,
                                  vasp_input_set=None,
                                  user_incar_settings=uis_ep,
                                  vasp_cmd=vasp_cmd,
                                  gamma_vasp_cmd=gamma_vasp_cmd,
                                  cust_args={}) for i in ["ep0", "ep1"]]
        # Create Firework links.
        fws = ep_fws + neb_fws
        links = {ep_fws[0]: [neb_fws[0]],
                 ep_fws[1]: [neb_fws[0]]}
        if neb_round > 1:
            for r in range(1, neb_round):
                links[neb_fws[r - 1]] = [neb_fws[r]]

        workflow = Workflow(fws, links_dict=links, name=wf_name)

    return workflow


def get_wf_neb_from_images(user_incar_settings, images=None,
                           wf_name=None, spec=None):
    """
    Get a CI-NEB workflow from given images.
    Workflow: NEB_1 -- NEB_2 - ... - NEB_n

    Args:
        user_incar_settings([dict]): Additional user_incar_settings
            corresponded with fw
        images ([Structure]): The image structures,
            if None then read from spec["images"]
        wf_name (str): some appropriate name for the workflow,
            also used to set running directory.
        spec (dict): user setting spec settings to overwrite spec_orig.

    Returns:
        Workflow
    """
    formula = images[0].composition.reduced_formula
    wf_name = wf_name or datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S-%f')

    if images is not None:
        for i in images:
            assert isinstance(i, Structure)
        images_dict = [s.as_dict() for s in images]
        spec = _update_spec_from_inputs(spec=spec, images=images_dict,
                                        wf_name=wf_name)
    else:
        spec = _update_spec_from_inputs(spec=spec, wf_name=wf_name)

    uis_neb = _get_incar("NEB", user_incar_settings)[2]
    neb_round = len(uis_neb)

    vasp_cmd = _get_mpi_command(spec, "std")
    gamma_vasp_cmd = _get_mpi_command(spec, "gam")

    fws = []
    for n in range(neb_round):
        fw = NEBFW(spec=spec,
                   name=formula,
                   neb_label=str(n + 1),
                   from_images=True,
                   user_incar_settings=uis_neb[n],
                   vasp_cmd=vasp_cmd,
                   gamma_vasp_cmd=gamma_vasp_cmd,
                   cust_args={})
        fws.append(fw)

    workflow = Workflow(fws, name=wf_name)

    return workflow


if __name__ == "__main__":
    pass

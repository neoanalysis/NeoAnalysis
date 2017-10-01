# -*- coding: utf-8 -*-
"""
Class for reading data from a Neurodata Without Borders (NWB) dataset

Depends on: h5py, nwb

Supported: Read, Write

Author: Andrew P. Davison, CNRS

"""

from __future__ import absolute_import
from __future__ import division
from itertools import chain
import shutil
import tempfile
from os.path import join

try:
    import h5py
except ImportError as err:
    HAVE_H5PY = False
    H5PY_ERR = err

from nwb import nwb_file
from nwb import nwb_utils
import quantities as pq
from ...neo.io.baseio import BaseIO
from ...neo.core import (Segment, SpikeTrain, Unit, Epoch, Event, AnalogSignal,
                      IrregularlySampledSignal, ChannelIndex, Block)

neo_extension = {"fs": {"neo": {
    "info": {
        "name": "Neo TimeSeries extension",
        "version": "0.1",
        "date": "2016-08-05",
        "author": "Andrew P. Davison",
        "contact": "andrew.davison@unic.cnrs-gif.fr",
        "description": ("Extension defining a new TimeSeries type, named 'MultiChannelTimeSeries'")
    },

    "schema": {
        "<MultiChannelTimeSeries>/": {
            "description": "Similar to ElectricalSeries, but without the restriction to volts",
            "merge": ["core:<TimeSeries>/"],
            "attributes": {
                "ancestry": {
                    "data_type": "text",
                    "dimensions": ["2"],
                    "value": ["TimeSeries", "MultiChannelTimeSeries"],
                    "const": True},
                "help": {
                    "data_type": "text",
                    "value": "A multi-channel time series",
                    "const": True}},
            "data": {
                "description": ("Multiple measurements are recorded at each point of time."),
                "dimensions": ["num_times", "num_channels"],
                "data_type": "float32"},
        }
    }
}}}



class NWBIO(BaseIO):
    """
    Class for "reading" experimental data from a .nwb file.
    """

    is_readable = True
    is_writable = True
    is_streameable = False
    supported_objects = [Block, Segment, AnalogSignal, IrregularlySampledSignal,
                         SpikeTrain, Epoch, Event, ChannelIndex, Unit]
    readable_objects  = supported_objects
    writeable_objects = supported_objects
    has_header = False
    name = 'NWB'
    description = 'This IO reads/writes experimental data from/to an .nwb dataset'
    extensions = ['nwb']
    mode = 'file'

    def __init__(self, filename) :
        """
        Arguments:
            filename : the filename
        """
        BaseIO.__init__(self, filename)
        self._extension_dir = tempfile.mkdtemp()
        extension_file = join(self._extension_dir, "nwb_neo_extension.py")
        with open(extension_file, "w") as fp:
            fp.write(str(neo_extension))
        self.extensions = [extension_file]

    def __del__(self):
        shutil.rmtree(self._extension_dir)


    def write_block(self, block, **kwargs):

        # todo: handle block.file_datetime, block.file_origin, block.index, block.annotations

        self._file = nwb_file.open(self.filename,
                          start_time=block.rec_datetime or "unknown",
                          mode="w",
                          identifier=block.name or "neo",
                          description=block.description or "no description",
                          core_spec="nwb_core.py", extensions=self.extensions,
                          default_ns="core", keep_original=False, auto_compress=True)

        for segment in block.segments:
            self._write_segment(segment)

        self._file.close()

    def _write_segment(self, segment):
        # Note that an NWB Epoch corresponds to a Neo Segment, not to a Neo Epoch.
        epoch = nwb_utils.create_epoch(self._file, segment.name,
                                       time_in_seconds(segment.t_start),
                                       time_in_seconds(segment.t_stop))
        for i, signal in enumerate(chain(segment.analogsignals, segment.irregularlysampledsignals)):
            self._write_signal(signal, epoch, i)
        for spiketrain in segment.spiketrains:
            pass
        for event in segment.events:
            pass
        for neo_epoch in segment.epochs:
            pass

    def _write_signal(self, signal, epoch, i):
        # for now, we assume that all signals should go in /acquisition
        # this could be modified using Neo annotations
        signal_name = signal.name or "signal{0}".format(i)
        ts_name = "{0}_{1}".format(signal.segment.name, signal_name)
        ts = self._file.make_group("<MultiChannelTimeSeries>", ts_name, path="/acquisition/timeseries")
        conversion, base_unit = _decompose_unit(signal.dimensionality)
        attributes = {"conversion": conversion,
                      "unit": base_unit,
                      "resolution": float('nan')}
        if isinstance(signal, AnalogSignal):
            ts.set_dataset("starting_time", time_in_seconds(signal.t_start),
                           attrs={"rate": float(signal.sampling_rate.rescale("Hz"))})
        elif isinstance(signal, IrregularlySampledSignal):
            ts.set_dataset("timestamps", signal.times.rescale('second').magnitude)
        else:
            raise TypeError("signal has type {0}, should be AnalogSignal or IrregularlySampledSignal".format(signal.__class__.__name__))
        ts.set_dataset("data", signal.magnitude,
                       attrs=attributes)
        ts.set_dataset("num_samples", signal.shape[0])  # this is supposed to be created automatically, but is not
        ts.set_attr("source", signal.name or "unknown")
        ts.set_attr("description", signal.description or "")

        nwb_utils.add_epoch_ts(epoch,
                               time_in_seconds(signal.segment.t_start),
                               time_in_seconds(signal.segment.t_stop),
                               signal_name,
                               ts)


def time_in_seconds(t):
    return float(t.rescale("second"))

def _decompose_unit(unit):
    """unit should be a Quantities Dimensionality object

    Returns (conversion, base_unit_str)
    """
    assert isinstance(unit, pq.dimensionality.Dimensionality)
    if len(unit) != 1:
        raise NotImplementedError("Compound units not yet supported")  # e.g. volt-metre
    uq, n = unit.items()[0]
    if n != 1:
        raise NotImplementedError("Compound units not yet supported")  # e.g. volt^2
    uq_def = uq.definition
    return float(uq_def.magnitude), uq_def.dimensionality.keys()[0].name

"""This file contains the module that writes the Gadget ICs to hdf5 files.
author: @mandarmk9
"""
import h5py
import numpy as np

class write(object):
    """Class which writes Gadget ICs to an hdf5 file"""
    def __init__(self, Pos, Vel, IDs, BoxSize, HubbleParam, MassTable,
        NumFilesPerSnapshot, NumPart_ThisFile, NumPart_Total, OmegaLambda, Omega0,
        Redshift, Time, filename):
        self.BoxSize = BoxSize
        self.HubbleParam = HubbleParam
        self.MassTable = MassTable
        self.NumFilesPerSnapshot = NumFilesPerSnapshot
        self.NumPart_ThisFile = NumPart_ThisFile
        self.NumPart_Total = NumPart_Total
        self.OmegaLambda = OmegaLambda
        self.Omega0 = Omega0
        self.Redshift = Redshift
        self.Time = Time
        self.Pos = Pos
        self.Vel = Vel
        self.IDs = IDs
        self.filename = filename

    def write_file(self):
        self.f = h5py.File(self.filename, 'w')
        self.f.create_group('Header')
        self.header = self.f['Header']

        self.header.attrs.create('BoxSize', self.BoxSize, dtype=float)
        # self.header.attrs.create('Flag_Cooling', self.Flag_Cooling, dtype=int)
        # self.header.attrs.create('Flag_Entropy_ICs', self.Flag_Entropy_ICs, dtype=int)
        # self.header.attrs.create('Flag_Sfr', self.Flag_Sfr, dtype=int)
        # self.header.attrs.create('Flag_StellarAge', self.Flag_StellarAge, dtype=int)
        self.header.attrs.create('HubbleParam', self.HubbleParam, dtype=float)
        self.header.attrs.create('MassTable', self.MassTable, dtype=float)
        self.header.attrs.create('NumFilesPerSnapshot', self.NumFilesPerSnapshot, dtype=int)
        self.header.attrs.create('NumPart_ThisFile', self.NumPart_ThisFile, dtype=int)
        self.header.attrs.create('NumPart_Total', self.NumPart_Total, dtype=int)
        # self.header.attrs.create('NumPart_Total_HighWord', self.NumPart_Total_HighWord, dtype=int)
        self.header.attrs.create('OmegaLambda', self.OmegaLambda, dtype=float)
        self.header.attrs.create('Omega0', self.Omega0, dtype=float)
        self.header.attrs.create('Redshift', self.Redshift, dtype=float)
        self.header.attrs.create('Time', self.Time, dtype=float)

        self.type1 = self.f.create_group('PartType1')
        self.type1.create_dataset('Coordinates', data=self.Pos)
        self.type1.create_dataset('ParticleIDs', data=self.IDs)
        self.type1.create_dataset('Velocities', data=self.Vel)

        self.f.close()

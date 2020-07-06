Extending TB2J
==============
In this section we show how to extend TB2J to interface with other first principles or similar codes and to write the output formats useful for codes such as spin dynamics.

Interface TB2J with other first principles or similar codes.
------------------------------------------------------------
To interface a DFT code with TB2J, one has only to implement a tight-binding-like model which has certain methods and properties implemented. TB2J make use of the duck type feature of python, thus any class which has these things can be plugged in. Then the object can be inputted to the TB2J.Exchange class.

The methods and properties of AbstractTB class is listed as below. 

.. autoclass:: TB2J.myTB.AbstractTB
   :members:

To pass the tight-binding-like model to the Exchange class is quite simple, here I take the sisl interface as an example ::

    # read hamiltonian using sisl
    fdf = sisl.get_sile(fdf_fname)
    H = fdf.read_hamiltonian()
    # wrap the hamiltonian to SislWrapper
    tbmodel = SislWrapper(H, spin=None)
    # pass to ExchangeNCL 
    exchange = ExchangeNCL(
        tbmodels=tbmodel,
        atoms=atoms,
        efermi=0.0,
        magnetic_elements=magnetic_elements,
        kmesh=kmesh,
        emin=emin,
        emax=emax,
        nz=nz,
        exclude_orbs=exclude_orbs,
        Rcut=Rcut,
        ne=ne,
        description=description)
    exchange.run()

In which the SislWrapper is a AbstractTB-like class which use sisl to read the Hamiltonian and overlap matrix from Siesta output. 

Extend the output to other formats
----------------------------------
The calculated magnetic interaction parameters, together with other informations, such as the atomic structure and some metadata, are saved in "SpinIO" object. By making use of it, it is easy to output the parameters to the file format needed. Some parameters, which cannot be calculated in TB2J can also be inputted so that they can be written to the files. The list of stored data is listed below, by using which it should be easy to write the output function as a member of the SpinIO class. A method write_some_format(path) can be implemented and called in the write_all method. Then the format is automatically written after the TB2J calculation.

.. autoclass:: TB2J.io_exchange.SpinIO
   :members:
   :undoc-members: atoms, spinat, charges, index_spin, distance_dict, has_exchange, exchange_Jdict, has_dmi, dmi_ddict, has_bilinear, Jani_dict

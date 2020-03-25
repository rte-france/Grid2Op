from grid2op.Exceptions import *
from grid2op.Space import SerializableSpace, GridObjects
from grid2op.Observation.CompleteObservation import CompleteObservation

class SerializableObservationSpace(SerializableSpace):
    """
    This class allows to serialize / de serialize the action space.

    It should not be used inside an Environment, as some functions of the action might not be compatible with
    the serialization, especially the checking of whether or not an Observation is legal or not.

    Attributes
    ----------

    observationClass: ``type``
        Type used to build the :attr:`SerializableActionSpace._template_act`

    _empty_obs: :class:`Observation`
        An instance of the "*observationClass*" provided used to provide higher level utilities

    """
    def __init__(self, gridobj, observationClass=CompleteObservation):
        """

        Parameters
        ----------
        gridobj: :class:`grid2op.Space.GridObjects`
            Representation of the objects in the powergrid.

        observationClass: ``type``
            Type of action used to build :attr:`Space.SerializableSpace._template_obj`

        """
        SerializableSpace.__init__(self, gridobj=gridobj, subtype=observationClass)

        self.observationClass = self.subtype
        self._empty_obs = self._template_obj

    @staticmethod
    def from_dict(dict_):
        """
        Allows the de-serialization of an object stored as a dictionnary (for example in the case of json saving).

        Parameters
        ----------
        dict_: ``dict``
            Representation of an Observation Space (aka SerializableObservationSpace) as a dictionnary.

        Returns
        -------
        res: :class:``SerializableObservationSpace``
            An instance of an action space matching the dictionnary.

        """
        tmp = SerializableSpace.from_dict(dict_)
        res = SerializableObservationSpace(gridobj=tmp,
                                           observationClass=tmp.subtype)
        return res

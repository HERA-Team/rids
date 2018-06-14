RF Interference Data System (RIDS)

File format to store data

A rids file is one that can be read and written by the rids_rw module if feature_module is None

The feature_module needs to, at a minimum, define:
    direct_attributes:  list of strings
    unit_attributes:  list of strings
    feature_components: list of strings
    feature_sets: dictionary
    and instantiate rids_rw.RidsReadWrite() -- which has its own direct_attributes
                                               and unit_attributes which the feature
                                               module can use

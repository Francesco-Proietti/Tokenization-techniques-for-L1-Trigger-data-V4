# Useful file used by train.py in order to pick the correct data-loading specified in the config file 

from src.data.event_particles_data_loading import EventPartL1TriggerDataModule
from src.data.jet_constituents_data_loading import JetConstL1TriggerDataModule
from src.data.event_jets_data_loading import EventJetsL1TriggerDataModule

DATA_REGISTRY = {
    "event_part": EventPartL1TriggerDataModule,
    "jet_const": JetConstL1TriggerDataModule,
    "event_jets": EventJetsL1TriggerDataModule,
}
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DatasetSource:
    address: str
    text_column: str
    label_column: str

all_data = []
all_data.append(DatasetSource(address="Rajarshi-Roy-research/Defactify_Text_Dataset", text_column="Text", label_column="Label_A"))
all_data.append(DatasetSource(address="andythetechnerd03/AI-human-text", text_column="text", label_column="generated"))
all_data.append(DatasetSource(address="Ateeqq/AI-and-Human-Generated-Text", text_column="abstract", label_column="label"))



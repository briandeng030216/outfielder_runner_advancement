# Data

The original data used to develop this project were supplied for a private recruiting exercise and are not included in this repository. The public project contains only a small synthetic dataset that demonstrates the expected schema and workflow.

`synthetic_sample.csv` does not contain real games, players, tracking observations, or assessment records. All identifiers and measurements were created for demonstration.

To use the project with another appropriately licensed dataset, prepare one row per advancement opportunity and match the fields documented in `data_dictionary.csv`. At minimum, the modeling workflow needs batted-ball measurements, catch and starting coordinates, runner and fielder speed estimates, the runner's starting base, a fielder identifier, and a binary advancement outcome.

Do not commit private or licensed raw data. Store it under `data/private/` or `data/raw/`, both of which are excluded by `.gitignore`.

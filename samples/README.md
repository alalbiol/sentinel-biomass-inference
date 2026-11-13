# Example imagery and reference predictions

Two 256×256 patches from the existing AI4DIS Spain benchmark are included: one mixed-forest and one coniferous example. Coordinates, date ranges, forest-type descriptors and source information are in `metadata.json`.

Contains modified Copernicus Sentinel data (2021), processed by AI4DIS from the Earth Engine collections `COPERNICUS/S1_GRD`, `COPERNICUS/S2_SR_HARMONIZED` and `COPERNICUS/S2_CLOUD_PROBABILITY`. Original data retain the [Copernicus Sentinel terms](https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice). This attribution applies to the sample imagery, not a claim that ESA endorses the predictions.

The existing benchmark used monthly per-orbit S1 medians and one full-coverage S2 scene per month ranked by patch-average cloud probability. Optical reflectance is raw DN and the s2cloudless channel is scaled to 0–255. Month 00 is January 2021 and month 11 is December 2021. Missing optical months, if present, are listed in `metadata.json`.

The sample GeoTIFFs were losslessly recompressed with DEFLATE; every decoded value and missing-month indicator was checked against the original project files. Inputs have four S1 bands and eleven S2 bands, on a common 10 m grid. No biomass ground-truth labels are included.

`expected/<model_id>/<chip_id>.npy` contains the original project model output, in Mg/ha, for every released model and both samples. These are regression-test references, not measurements or accuracy labels. The reference predictions may be reused under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), with attribution to the four project authors listed in the root README.
